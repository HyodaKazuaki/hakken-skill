"""Graph-branch scorers.

Both scorers expose: score(u, v, exclude=None, only=None) -> (dict token -> value, support)
where token 'R>' means (u, R, v) and 'R<' means (v, R, u). `exclude`/`only` are sets of
(s, r, o) triples defining perturbed graph views (used by PHELInE-style explanations).

* FeatureScorer  - temporal network features (Science4Cast-style: degrees, common
  neighbours, Adamic-Adar, resource allocation, Jaccard, recency-weighted paths, growth,
  node age, 3-hop paths) + relation-pattern / profile label features, combined by a
  logistic model fitted in seconds on the graph's OWN history: snapshot at T-W ->
  which triples appeared in (T-W, T]. Output = probability that the triple is reported
  within the next W time units (a lower bound under positive-unlabeled data).
* HeuristicScorer - fixed-weight path score (no fitting at all); optional Platt scaling.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from .common import log, pkey, sigmoid
from .graph import TKG, LabelModel, candidate_pairs

GENERIC = ["cn", "adamic_adar", "resource_alloc", "jaccard", "temporal_paths",
           "log_deg_min", "log_deg_max", "recent_cn", "growth_min", "growth_max",
           "log_age_min", "log_age_max", "paths3", "support"]
LABELF = ["pattern", "profile", "log_prior", "pattern_x_paths"]


def _only_adj(only):
    if only is None:
        return None
    adj = defaultdict(set)
    for s, _r, o in only:
        adj[s].add(o)
        adj[o].add(s)
    return adj


class _Base:
    def __init__(self, g, lm=None, hops=2, nbr_cap=40, mu=0.3, gamma3=0.2):
        self.g = g
        self.lm = lm or LabelModel(g)
        self.hops = hops
        self.nbr_cap = nbr_cap
        self.mu = mu
        self.gamma3 = gamma3
        self._capcache = {}

    def _nbrs(self, x, adj):
        return adj.get(x, set()) if adj is not None else self.g.nbrs.get(x, set())

    def _capped(self, x, adj):
        if adj is not None:
            return adj.get(x, set())
        c = self._capcache.get(x)
        if c is None:
            ns = self.g.nbrs.get(x, set())
            c = set(sorted(ns, key=lambda n: -self.g.weight(x, n))[: self.nbr_cap]) if len(ns) > self.nbr_cap else set(ns)
            self._capcache[x] = c
        return c

    def _paths(self, u, v, exclude, only, adj):
        """Shared path statistics on a (possibly perturbed) view."""
        g = self.g
        st = dict(cn=0, aa=0.0, ra=0.0, tw=0.0, recent=0, p3=0.0, sup=0.0)
        typ, twt = Counter(), 0.0
        nu, nv = self._nbrs(u, adj), self._nbrs(v, adj)
        union = len(nu | nv) or 1
        for n in nu & nv:
            if n in (u, v):
                continue
            w1, w2 = g.weight(u, n, exclude, only), g.weight(n, v, exclude, only)
            if w1 <= 0 or w2 <= 0:
                continue
            d = g.deg.get(n, 1)
            st["cn"] += 1
            st["aa"] += 1.0 / math.log(2 + d)
            st["ra"] += 1.0 / max(1, d)
            c = math.sqrt(w1 * w2) / math.log(2 + d)
            st["tw"] += c
            y1, y2 = g.first_year(u, n, exclude, only), g.first_year(n, v, exclude, only)
            if y1 is not None and y2 is not None and min(y1, y2) > g.T - 3:
                st["recent"] += 1
            st["sup"] += math.log1p(sum(g.support.get(t[:3], 1) for t in g.triples_between(u, n))) \
                + math.log1p(sum(g.support.get(t[:3], 1) for t in g.triples_between(n, v)))
            for lab, p in self.lm.pattern_dist(g.tokens(u, n, exclude, only), g.tokens(n, v, exclude, only)).items():
                typ[lab] += c * p
            twt += c
        if self.hops >= 3:
            for a in self._capped(u, adj) - {v}:
                na = self._nbrs(a, adj)
                for b in self._capped(v, adj) - {u}:
                    if a == b or b not in na:
                        continue
                    w1, w2, w3 = g.weight(u, a, exclude, only), g.weight(a, b, exclude, only), g.weight(b, v, exclude, only)
                    if w1 <= 0 or w2 <= 0 or w3 <= 0:
                        continue
                    st["p3"] += (w1 * w2 * w3) ** (1.0 / 3) / (
                        math.log(2 + g.deg.get(a, 0)) * math.log(2 + g.deg.get(b, 0)))
        st["jac"] = st["cn"] / union
        typ = {k: v / twt for k, v in typ.items()} if twt > 0 else {}
        return st, typ


class HeuristicScorer(_Base):
    """score = existence * label distribution (no fitted parameters)."""
    kind = "heuristic"
    is_prob = False

    def score(self, u, v, exclude=None, only=None):
        st, typ = self._paths(u, v, exclude, only, _only_adj(only))
        e = st["tw"] + self.gamma3 * st["p3"]
        if e <= 0:
            return {}, 0.0
        prof = self.lm.profile(u, v) or self.lm.prior
        t = {k: (1 - self.mu) * p for k, p in typ.items()} if typ else {}
        w = self.mu if typ else 1.0
        for k, p in prof.items():
            t[k] = t.get(k, 0.0) + w * p
        return {k: e * p for k, p in t.items() if p > 0}, e


class FeatureScorer(_Base):
    """Logistic model over temporal network features (parameters from fit_feature_model)."""
    kind = "features"
    is_prob = True

    def __init__(self, g, params, lm=None, **kw):
        super().__init__(g, lm, hops=params.get("hops", 2), nbr_cap=params.get("nbr_cap", 40), **kw)
        self.params = params
        self.labels = params["labels"]

    def featurize(self, u, v, exclude=None, only=None):
        g = self.g
        st, typ = self._paths(u, v, exclude, only, _only_adj(only))
        du, dv = g.deg.get(u, 0), g.deg.get(v, 0)
        gu, gv = g.growth(u), g.growth(v)
        au = g.T - g.node_first.get(u, g.T) + 1
        av = g.T - g.node_first.get(v, g.T) + 1
        gen = [math.log1p(st["cn"]), st["aa"], st["ra"], st["jac"], math.log1p(st["tw"]),
               math.log1p(min(du, dv)), math.log1p(max(du, dv)), math.log1p(st["recent"]),
               min(gu, gv), max(gu, gv), math.log1p(min(au, av)), math.log1p(max(au, av)),
               math.log1p(st["p3"]), math.log1p(st["sup"])]
        return gen, typ, self.lm.profile(u, v), st

    def label_feats(self, lab, typ, prof, gen):
        t = typ.get(lab, 0.0)
        return [t, prof.get(lab, 0.0), math.log(self.lm.prior.get(lab, 1e-6) + 1e-6), t * gen[4]]

    def score(self, u, v, exclude=None, only=None):
        gen, typ, prof, st = self.featurize(u, v, exclude, only)
        if st["cn"] == 0 and st["p3"] == 0:
            return {}, 0.0
        P = self.params
        mg, sg, ml, sl = P["mean_g"], P["std_g"], P["mean_l"], P["std_l"]
        zg = [(x - m) / s for x, m, s in zip(gen, mg, sg)]
        base = P["intercept"] + sum(w * x for w, x in zip(P["w_g"], zg))
        out = {}
        for lab in self.labels:
            lf = self.label_feats(lab, typ, prof, gen)
            z = base + P["bias"].get(lab[:-1], 0.0) + sum(
                w * (x - m) / s for w, x, m, s in zip(P["w_l"], lf, ml, sl))
            out[lab] = sigmoid(z)
        return out, st["tw"]


# --------------------------------------------------------------------------- #
# Fitting (seconds, CPU, stdlib): weighted logistic regression via Newton-Raphson
# --------------------------------------------------------------------------- #


def _solve(A, b):
    """Gaussian elimination with partial pivoting (A is small, dense)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < 1e-12:
            continue
        M[c], M[piv] = M[piv], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for c in range(n - 1, -1, -1):
        if abs(M[c][c]) < 1e-12:
            continue
        x[c] = (M[c][n] - sum(M[c][k] * x[k] for k in range(c + 1, n))) / M[c][c]
    return x


def _fit_logreg(rows, n_dense, labels, l2=1.0, l2_bias=3.0, iters=12):
    """rows: list of (dense_x incl. const at index 0, label_idx, y, weight)."""
    L = len(labels)
    D = n_dense + L
    beta = [0.0] * D
    for it in range(iters):
        grad = [0.0] * D
        H = [[0.0] * D for _ in range(D)]
        loss = 0.0
        for x, li, y, w in rows:
            z = sum(b * xi for b, xi in zip(beta, x)) + beta[n_dense + li]
            p = sigmoid(z)
            loss -= w * (y * math.log(p + 1e-12) + (1 - y) * math.log(1 - p + 1e-12))
            d = w * (p - y)
            h = w * p * (1 - p)
            bi = n_dense + li
            for i in range(n_dense):
                grad[i] += d * x[i]
                hx = h * x[i]
                Hi = H[i]
                for j in range(i, n_dense):
                    Hi[j] += hx * x[j]
                Hi[bi] += hx
            grad[bi] += d
            H[bi][bi] += h
        for i in range(D):  # symmetrise + ridge (no penalty on the constant)
            for j in range(i + 1, D):
                H[j][i] = H[i][j]
            lam = 0.0 if i == 0 else (l2 if i < n_dense else l2_bias)
            grad[i] += lam * beta[i]
            H[i][i] += lam + 1e-9
        step = _solve(H, grad)
        beta = [b - s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < 1e-6:
            break
    return beta, loss


def fit_feature_model(first, cutoff, window, support=None, types=None, tau=10.0, hops=2,
                      nbr_cap=40, seeds=None, seed_mode="any", hub_cap=1000,
                      max_train_pairs=20000, neg_per_pos=30, max_rows=60000, seed=7):
    """Fit on transition (cutoff - window) -> (cutoff - window, cutoff]."""
    rng = random.Random(seed)
    t0 = cutoff - window
    g0 = TKG(first, cutoff=t0, tau=tau, support=support, types=types)
    if not g0.first:
        raise ValueError("no triples at or before %s; choose a later cutoff or smaller window" % t0)
    fs = FeatureScorer(g0, {"labels": [], "hops": hops, "nbr_cap": nbr_cap})
    labels = fs.lm.labels
    # one intercept per relation NAME: direction is carried by the (orientation-symmetric)
    # pattern/profile features, so scores do not depend on how a pair happens to be ordered
    rels = sorted({l[:-1] for l in labels})
    lidx = {l: rels.index(l[:-1]) for l in labels}
    newpos = defaultdict(set)
    for (s, r, o), y in first.items():
        if t0 < y <= cutoff and s in g0.nbrs and o in g0.nbrs and not g0.connected(s, o):
            a, b = pkey(s, o)
            newpos[(a, b)].add(r + (">" if s == a else "<"))
    pairs = candidate_pairs(g0, seeds, seed_mode, False, hub_cap)
    pos_pairs = [p for p in pairs if p in newpos]
    neg_pairs = [p for p in pairs if p not in newpos]
    qp = 1.0
    room = max(0, max_train_pairs - len(pos_pairs))
    if len(neg_pairs) > room:
        qp = room / len(neg_pairs) if neg_pairs else 1.0
        neg_pairs = rng.sample(neg_pairs, room)
    n_pos_rows = sum(len(newpos[p] & set(labels)) for p in pos_pairs)
    if n_pos_rows < 10:
        raise ValueError("only %d positive examples in (%s, %s]; use a larger --window or more data"
                         % (n_pos_rows, t0, cutoff))
    n_neg_rows = (len(pos_pairs) + len(neg_pairs)) * len(labels) - n_pos_rows
    ql = min(1.0, neg_per_pos * n_pos_rows / max(1, n_neg_rows), max(1, max_rows - n_pos_rows) / max(1, n_neg_rows))
    raw = []  # (gen, labfeat, li, y, w)
    for p, is_pos in [(p, True) for p in pos_pairs] + [(p, False) for p in neg_pairs]:
        a, b = p
        gen, typ, prof, st = fs.featurize(a, b)
        if st["cn"] == 0 and st["p3"] == 0:
            continue
        for lab in labels:
            y = 1 if (is_pos and lab in newpos[p]) else 0
            if y == 0 and rng.random() > ql:
                continue
            w = 1.0 if y else 1.0 / (ql * (1.0 if is_pos else qp))
            raw.append((gen, fs.label_feats(lab, typ, prof, gen), lidx[lab], y, w))
    npos = sum(r[3] for r in raw)
    if npos < 10:
        raise ValueError("too few positives with structural support (%d)" % npos)

    def stats(vals):
        n = len(vals)
        m = sum(vals) / n
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / n) or 1.0
        return m, s

    mg, sg, ml, sl = [], [], [], []
    for j in range(len(GENERIC)):
        m, s = stats([r[0][j] for r in raw])
        mg.append(m)
        sg.append(s)
    for j in range(len(LABELF)):
        m, s = stats([r[1][j] for r in raw])
        ml.append(m)
        sl.append(s)
    rows = []
    for gen, lf, li, y, w in raw:
        x = [1.0] + [(v - m) / s for v, m, s in zip(gen, mg, sg)] + [(v - m) / s for v, m, s in zip(lf, ml, sl)]
        rows.append((x, li, y, w))
    n_dense = 1 + len(GENERIC) + len(LABELF)
    beta, loss = _fit_logreg(rows, n_dense, rels)
    params = {
        "model": "features-logreg-v1", "fit_cutoff": cutoff, "window": window, "tau": tau,
        "hops": hops, "nbr_cap": nbr_cap, "labels": labels,
        "generic_features": GENERIC, "label_features": LABELF,
        "intercept": beta[0], "w_g": beta[1:1 + len(GENERIC)],
        "w_l": beta[1 + len(GENERIC):n_dense],
        "bias": {r: beta[n_dense + i] for i, r in enumerate(rels)},
        "mean_g": mg, "std_g": sg, "mean_l": ml, "std_l": sl,
        "train": {"pairs_pos": len(pos_pairs), "pairs_neg_sampled": len(neg_pairs), "pair_rate": qp,
                  "label_neg_rate": ql, "rows": len(rows), "positives": npos, "weighted_logloss": loss},
    }
    log("[fit] features model on (%s, %s]: %d rows, %d positives, %d labels" % (t0, cutoff, len(rows), npos, len(labels)))
    return params


def describe_weights(params):
    items = list(zip(params["generic_features"], params["w_g"])) + list(zip(params["label_features"], params["w_l"]))
    return sorted(items, key=lambda x: -abs(x[1]))


# --------------------------------------------------------------------------- #
# Platt calibration for the heuristic scorer
# --------------------------------------------------------------------------- #


def fit_platt(xs, ys, iters=50, l2=1e-3):
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = 0.0
        haa = hab = hbb = l2
        for x, y in zip(xs, ys):
            p = sigmoid(a * x + b)
            d = p - y
            ga += d * x
            gb += d
            w = p * (1 - p)
            haa += w * x * x
            hab += w * x
            hbb += w
        ga += l2 * a
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-7:
            break
    return a, b


def calibrate_heuristic(first, cutoff, window, tau=10.0, hops=2, nbr_cap=40, mu=0.3,
                        seeds=None, seed_mode="any", hub_cap=1000, max_pairs=None, support=None):
    t0 = cutoff - window
    g0 = TKG(first, cutoff=t0, tau=tau, support=support)
    sc = HeuristicScorer(g0, hops=hops, nbr_cap=nbr_cap, mu=mu)
    rng = random.Random(7)
    pos, neg = defaultdict(list), defaultdict(list)
    for a, b in candidate_pairs(g0, seeds, seed_mode, False, hub_cap, max_pairs):
        scores, e = sc.score(a, b)
        for tok, val in scores.items():
            s, r, o = (a, tok[:-1], b) if tok[-1] == ">" else (b, tok[:-1], a)
            y = first.get((s, r, o))
            (pos if (y is not None and t0 < y <= cutoff) else neg)[r].append(math.log1p(val))
    params, allx, ally = {}, [], []
    for r in set(pos) | set(neg):
        p, n = pos[r], neg[r]
        if len(n) > 20000:
            n = rng.sample(n, 20000)
        allx += p + n[:5000]
        ally += [1] * len(p) + [0] * len(n[:5000])
        if len(p) >= 5 and len(n) >= 5:
            params[r] = fit_platt(p + n, [1] * len(p) + [0] * len(n))
    pooled = fit_platt(allx, ally) if allx and any(ally) else None
    log("[calibrate] heuristic Platt on (%s, %s]: %d relations, %d positives" % (t0, cutoff, len(params), sum(ally)))
    return {"per_relation": params, "pooled": pooled}


def apply_platt(cal, r, val):
    ab = cal["per_relation"].get(r) or cal["pooled"]
    if ab is None:
        return None
    return sigmoid(ab[0] * math.log1p(val) + ab[1])
