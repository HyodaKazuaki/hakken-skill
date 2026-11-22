#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hakken_kg.py - a dependency-free, Hakken-style knowledge-prediction toolkit.

It reproduces the *workflow* of Hakken (Besold et al., arXiv:2609.04494) on any
temporal knowledge graph of (subject, relation, object, year) triples:

  stats     inspect a triple file (size, years, relation/hub distribution)
  clean     normalise entities/relations, earliest-occurrence dedupe,
            polarity-conflict reconciliation, rare-relation pruning (paper App. E)
  predict   score as-yet-unobserved (s, r, o) hypotheses with a temporal,
            relation-aware path model; optional Platt calibration on a
            held-out recent window (paper Sec. 2 / App. B, lightweight)
  backtest  time-sliced evaluation: interval recall vs. a random-empirical-score
            null baseline + label nDCG over disjoint future bins (paper Sec. 4.3)
  explain   PHELInE-style explanations: enumerate s-o paths, score sufficiency
            and necessity by re-querying the scorer on perturbed graphs,
            rerank for diversity; optional latent (predicted) paths (App. D)
  select    hypothesis-selection pipeline: novelty filter, recency, top-k per
            entity, shortest-path length (paper App. F.2)

The scorer is NOT THiGERLLM. It is a transparent surrogate that keeps the
same inductive biases (temporal accumulation of evidence, multi-hop structure,
relation-type patterns, positive-unlabeled framing). For the full model see
https://github.com/SonyResearch/HakkenOSS (referenced in the paper).

Only the Python 3.8+ standard library is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict, deque

# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

ALIASES = {
    "subject": ["subject", "s", "head", "source", "src", "subject_name"],
    "relation": ["relation", "r", "predicate", "rel", "relation_type", "edge_type"],
    "object": ["object", "o", "tail", "target", "dst", "object_name"],
    "year": ["year", "time", "date", "timestamp", "pub_year", "first_year"],
}


def _log(msg):
    print(msg, file=sys.stderr)


def _open_out(path):
    if path in (None, "-"):
        return sys.stdout
    return open(path, "w", newline="", encoding="utf-8")


def _parse_year(v):
    v = str(v).strip()
    if re.fullmatch(r"-?\d+", v):  # plain integer: year, yyyymm, or any ordinal time step
        return int(v)
    m = re.search(r"(\d{4})", v)
    if not m:
        raise ValueError("cannot parse year from %r" % (v,))
    return int(m.group(1))


def read_triples(path):
    """Return list of (s, r, o, year, extra_dict)."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        head = fh.readline()
        fh.seek(0)
        delim = "\t" if head.count("\t") > head.count(",") else ","
        rd = csv.DictReader(fh, delimiter=delim)
        cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
        mapping = {}
        for key, names in ALIASES.items():
            for n in names:
                if n in cols:
                    mapping[key] = cols[n]
                    break
            if key not in mapping:
                raise SystemExit(
                    "column for %r not found in %s (have: %s)"
                    % (key, path, ", ".join(rd.fieldnames or []))
                )
        used = set(mapping.values())
        out = []
        bad = 0
        for row in rd:
            try:
                s = (row[mapping["subject"]] or "").strip()
                r = (row[mapping["relation"]] or "").strip()
                o = (row[mapping["object"]] or "").strip()
                y = _parse_year(row[mapping["year"]])
            except (ValueError, KeyError):
                bad += 1
                continue
            if not s or not r or not o:
                bad += 1
                continue
            extra = {k: v for k, v in row.items() if k not in used and k is not None}
            out.append((s, r, o, y, extra))
    if bad:
        _log("[read] skipped %d malformed rows" % bad)
    return out


def write_rows(path, fieldnames, rows):
    fh = _open_out(path)
    w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow(row)
    if fh is not sys.stdout:
        fh.close()


def pkey(a, b):
    return (a, b) if a <= b else (b, a)


def inv(tok):
    """Invert a directed relation token ('R>' <-> 'R<')."""
    return tok[:-1] + ("<" if tok[-1] == ">" else ">")


def tok_to_triple(u, v, tok):
    rel = tok[:-1]
    return (u, rel, v) if tok[-1] == ">" else (v, rel, u)


# --------------------------------------------------------------------------- #
# Temporal knowledge graph
# --------------------------------------------------------------------------- #


class TKG:
    """Snapshot of the literature graph up to (and including) `cutoff`.

    Only the first observation of each triple is kept (as in the paper).
    Edge weight between two entities = sum over their triples of
    0.5 + 0.5 * exp(-(T - year) / tau): evidence accumulates, recent evidence
    counts more, old evidence never vanishes.
    """

    def __init__(self, triples, cutoff=None, tau=10.0):
        first = {}
        for s, r, o, y, *_ in triples:
            if cutoff is not None and y > cutoff:
                continue
            if s == o:
                continue
            k = (s, r, o)
            if k not in first or y < first[k]:
                first[k] = y
        self.first = first
        self.T = cutoff if cutoff is not None else max(first.values(), default=0)
        self.tau = float(tau)
        self.pair = defaultdict(list)
        self.nbrs = defaultdict(set)
        self.out_rel = defaultdict(Counter)
        self.in_rel = defaultdict(Counter)
        self.rel_count = Counter()
        self.node_years = defaultdict(list)
        for (s, r, o), y in first.items():
            self.pair[pkey(s, o)].append((s, r, o, y))
            self.nbrs[s].add(o)
            self.nbrs[o].add(s)
            self.out_rel[s][r] += 1
            self.in_rel[o][r] += 1
            self.rel_count[r] += 1
            self.node_years[s].append(y)
            self.node_years[o].append(y)
        self.relations = sorted(self.rel_count)
        self.deg = {n: len(v) for n, v in self.nbrs.items()}
        self._wcache = {}

    def decay(self, y):
        return 0.5 + 0.5 * math.exp(-max(0, self.T - y) / self.tau)

    def connected(self, a, b):
        return pkey(a, b) in self.pair

    def weight(self, a, b, exclude=None, only=None):
        if exclude is None and only is None:
            k = pkey(a, b)
            w = self._wcache.get(k)
            if w is None:
                w = sum(self.decay(t[3]) for t in self.pair.get(k, ()))
                self._wcache[k] = w
            return w
        w = 0.0
        for t in self.pair.get(pkey(a, b), ()):
            tr = t[:3]
            if exclude is not None and tr in exclude:
                continue
            if only is not None and tr not in only:
                continue
            w += self.decay(t[3])
        return w

    def tokens(self, a, b, exclude=None, only=None):
        """Directed relation tokens on edge a-b, seen from a."""
        out = []
        for s, r, o, _y in self.pair.get(pkey(a, b), ()):
            tr = (s, r, o)
            if exclude is not None and tr in exclude:
                continue
            if only is not None and tr not in only:
                continue
            out.append(r + (">" if s == a else "<"))
        return out

    def triples_between(self, a, b):
        return [t for t in self.pair.get(pkey(a, b), ())]

    def recency(self, n):
        ys = self.node_years.get(n)
        if not ys:
            return None
        m = statistics.median(ys)
        return int(m) if m == int(m) else round(m, 1)

    def shortest_path_len(self, a, b, max_len=6):
        if a not in self.nbrs or b not in self.nbrs:
            return None
        if a == b:
            return 0
        seen = {a}
        frontier = [a]
        for d in range(1, max_len + 1):
            nxt = []
            for x in frontier:
                for y in self.nbrs[x]:
                    if y == b:
                        return d
                    if y not in seen:
                        seen.add(y)
                        nxt.append(y)
            frontier = nxt
            if not frontier:
                break
        return None


# --------------------------------------------------------------------------- #
# Relation-pattern mining (label-type model)
# --------------------------------------------------------------------------- #


def mine_patterns(g, max_common=100, seed=13):
    """Count which relation label closes each 2-hop pattern (t1, t2).

    conf(l | t1, t2) ~= P(label l on u-v | u -t1- n -t2- v, u-v connected).
    Counted from both viewpoints so lookups are direction-consistent.
    """
    rng = random.Random(seed)
    closed = Counter()
    total = Counter()
    for (a, b), trips in g.pair.items():
        common = g.nbrs[a] & g.nbrs[b]
        if not common:
            continue
        if len(common) > max_common:
            common = rng.sample(sorted(common), max_common)
        for u, v in ((a, b), (b, a)):
            labs = set(r + (">" if s == u else "<") for s, r, o, _ in trips)
            for n in common:
                for t1 in g.tokens(u, n):
                    for t2 in g.tokens(n, v):
                        total[(t1, t2)] += 1
                        for lab in labs:
                            closed[(t1, t2, lab)] += 1
    return closed, total


class Scorer:
    """Transparent surrogate for a temporal KG relation predictor.

    existence(u, v) = sum_{2-hop paths} sqrt(w1 w2) / log(2 + deg n)
                    + gamma3 * sum_{3-hop paths} (w1 w2 w3)^(1/3) / (log(2+deg a) log(2+deg b))
    type(l | u, v)  = (1 - mu) * path-pattern label distribution + mu * node-profile prior
    score(u, l, v)  = existence * type
    """

    def __init__(self, g, hops=2, nbr_cap=40, mu=0.3, beta=2.0, gamma3=0.2,
                 max_common=100, seed=13):
        self.g = g
        self.hops = hops
        self.nbr_cap = nbr_cap
        self.mu = mu
        self.beta = beta
        self.gamma3 = gamma3
        closed, total = mine_patterns(g, max_common=max_common, seed=seed)
        lab_count = Counter()
        for (s, r, o), _y in g.first.items():
            lab_count[r + ">"] += 1
            lab_count[r + "<"] += 1
        z = sum(lab_count.values()) or 1
        self.prior = {k: v / z for k, v in lab_count.items()}
        self.labels = sorted(self.prior)
        by_pat = defaultdict(dict)
        for (t1, t2, lab), c in closed.items():
            by_pat[(t1, t2)][lab] = c
        self.conf = {}
        for pat, tot in total.items():
            d = by_pat.get(pat, {})
            self.conf[pat] = {
                lab: (d.get(lab, 0) + beta * p) / (tot + beta)
                for lab, p in self.prior.items()
                if d.get(lab, 0) > 0 or p > 0.01
            }
        self._capcache = {}

    # -- helpers ------------------------------------------------------------ #
    def _nbrs(self, x, only_adj):
        if only_adj is not None:
            return only_adj.get(x, set())
        return self.g.nbrs.get(x, set())

    def _capped(self, x, only_adj):
        if only_adj is not None:
            return only_adj.get(x, set())
        c = self._capcache.get(x)
        if c is None:
            ns = self.g.nbrs.get(x, set())
            if len(ns) > self.nbr_cap:
                c = set(sorted(ns, key=lambda n: -self.g.weight(x, n))[: self.nbr_cap])
            else:
                c = set(ns)
            self._capcache[x] = c
        return c

    def profile(self, u, v):
        g = self.g
        ou, iu = g.out_rel.get(u, Counter()), g.in_rel.get(u, Counter())
        ov, iv = g.out_rel.get(v, Counter()), g.in_rel.get(v, Counter())
        nu = (sum(ou.values()) + sum(iu.values())) or 1
        nv = (sum(ov.values()) + sum(iv.values())) or 1
        d = {}
        for r in g.relations:
            a = math.sqrt((ou[r] / nu) * (iv[r] / nv))
            b = math.sqrt((iu[r] / nu) * (ov[r] / nv))
            if a > 0:
                d[r + ">"] = a
            if b > 0:
                d[r + "<"] = b
        z = sum(d.values())
        if z <= 0:
            return dict(self.prior)
        return {k: v / z for k, v in d.items()}

    def _pattern_dist(self, t1s, t2s):
        acc = Counter()
        n = 0
        for t1 in t1s:
            for t2 in t2s:
                d = self.conf.get((t1, t2))
                if d is None:
                    d = self.prior
                for lab, p in d.items():
                    acc[lab] += p
                n += 1
        if n == 0:
            return {}
        return {k: v / n for k, v in acc.items()}

    # -- main --------------------------------------------------------------- #
    def score(self, u, v, exclude=None, only=None, parts=False):
        """Return (dict label->score, existence, list of path contributions)."""
        g = self.g
        only_adj = None
        if only is not None:
            only_adj = defaultdict(set)
            for s, _r, o in only:
                only_adj[s].add(o)
                only_adj[o].add(s)
        e = 0.0
        typ = Counter()
        twt = 0.0
        contrib = []
        nu = self._nbrs(u, only_adj)
        nv = self._nbrs(v, only_adj)
        for n in nu & nv:
            if n in (u, v):
                continue
            w1 = g.weight(u, n, exclude, only)
            w2 = g.weight(n, v, exclude, only)
            if w1 <= 0 or w2 <= 0:
                continue
            c = math.sqrt(w1 * w2) / math.log(2 + g.deg.get(n, 0))
            e += c
            d = self._pattern_dist(g.tokens(u, n, exclude, only), g.tokens(n, v, exclude, only))
            for lab, p in d.items():
                typ[lab] += c * p
            twt += c
            if parts:
                contrib.append(((u, n, v), c))
        if self.hops >= 3:
            au = self._capped(u, only_adj) - {v}
            bv = self._capped(v, only_adj) - {u}
            for a in au:
                na = self._nbrs(a, only_adj)
                for b in bv:
                    if a == b or b not in na:
                        continue
                    w1 = g.weight(u, a, exclude, only)
                    w2 = g.weight(a, b, exclude, only)
                    w3 = g.weight(b, v, exclude, only)
                    if w1 <= 0 or w2 <= 0 or w3 <= 0:
                        continue
                    c = self.gamma3 * (w1 * w2 * w3) ** (1.0 / 3) / (
                        math.log(2 + g.deg.get(a, 0)) * math.log(2 + g.deg.get(b, 0))
                    )
                    e += c
                    if parts:
                        contrib.append(((u, a, b, v), c))
        prof = self.profile(u, v)
        if twt > 0:
            t = {k: (1 - self.mu) * val / twt for k, val in typ.items()}
            for k, val in prof.items():
                t[k] = t.get(k, 0.0) + self.mu * val
        else:
            t = prof
        return {k: e * p for k, p in t.items() if p > 0}, e, contrib


# --------------------------------------------------------------------------- #
# Candidate generation, scoring and calibration
# --------------------------------------------------------------------------- #


def read_list(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def candidate_pairs(g, seeds=None, seed_mode="any", include_connected=False,
                    hub_cap=1000, max_pairs=None):
    pairs = set()
    if seeds is not None:
        seeds = [s for s in seeds if s in g.nbrs]
    if seeds is not None and seed_mode == "both":
        ss = sorted(set(seeds))
        for i, a in enumerate(ss):
            for b in ss[i + 1:]:
                if include_connected or not g.connected(a, b):
                    pairs.add((a, b))
    else:
        sources = seeds if seeds is not None else list(g.nbrs)
        for u in sources:
            for n in g.nbrs[u]:
                if g.deg[n] > hub_cap:
                    continue
                for v in g.nbrs[n]:
                    if v == u:
                        continue
                    if not include_connected and g.connected(u, v):
                        continue
                    pairs.add(pkey(u, v))
                    if max_pairs and len(pairs) >= max_pairs:
                        return sorted(pairs)
    return sorted(pairs)


def score_pairs(scorer, pairs, include_connected=False, min_score=0.0):
    """Yield (subject, relation, object, score, pair_existence)."""
    g = scorer.g
    for a, b in pairs:
        sc, e, _ = scorer.score(a, b)
        if e <= 0:
            continue
        for tok, val in sc.items():
            if val <= min_score:
                continue
            s, r, o = tok_to_triple(a, b, tok)
            if include_connected and (s, r, o) in g.first:
                continue
            yield s, r, o, val, e


def _fit_platt(xs, ys, iters=50, l2=1e-3):
    """1-D logistic regression p = sigmoid(a*x + b) via Newton's method."""
    a, b = 1.0, 0.0
    for _ in range(iters):
        ga = gb = 0.0
        haa = hab = hbb = l2
        for x, y in zip(xs, ys):
            z = max(-30.0, min(30.0, a * x + b))
            p = 1.0 / (1.0 + math.exp(-z))
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
        a -= da
        b -= db
        if abs(da) + abs(db) < 1e-7:
            break
    return a, b


def calibrate(triples, cutoff, window, args, seeds):
    """Fit per-relation Platt scalers: score at (cutoff - window) -> appears in (cutoff-window, cutoff]."""
    t0 = cutoff - window
    g0 = TKG(triples, cutoff=t0, tau=args.tau)
    full = TKG(triples, cutoff=cutoff, tau=args.tau)
    sc0 = Scorer(g0, hops=args.hops, nbr_cap=args.nbr_cap, mu=args.mu)
    pairs = candidate_pairs(g0, seeds, args.seed_mode, args.include_connected,
                            args.hub_cap, args.max_pairs)
    rng = random.Random(7)
    pos = defaultdict(list)
    neg = defaultdict(list)
    for s, r, o, val, _e in score_pairs(sc0, pairs, args.include_connected):
        x = math.log1p(val)
        y = full.first.get((s, r, o))
        if y is not None and t0 < y <= cutoff:
            pos[r].append(x)
        else:
            neg[r].append(x)
    params = {}
    allx, ally = [], []
    for r in set(pos) | set(neg):
        p, n = pos[r], neg[r]
        if len(n) > 20000:
            n = rng.sample(n, 20000)
        allx += p + n[:5000]
        ally += [1] * len(p) + [0] * len(n[:5000])
        if len(p) >= 5 and len(n) >= 5:
            params[r] = _fit_platt(p + n, [1] * len(p) + [0] * len(n))
    pooled = _fit_platt(allx, ally) if allx and any(ally) else None
    _log("[calibrate] window (%d, %d]: %d relations fitted individually, %d positives total"
         % (t0, cutoff, len(params), sum(ally)))
    return params, pooled


def apply_platt(params, pooled, r, val):
    ab = params.get(r) or pooled
    if ab is None:
        return None
    z = max(-30.0, min(30.0, ab[0] * math.log1p(val) + ab[1]))
    return 1.0 / (1.0 + math.exp(-z))


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def cmd_stats(args):
    tr = read_triples(args.triples)
    g = TKG(tr, cutoff=args.cutoff, tau=args.tau)
    years = Counter(g.first.values())
    print("# Graph statistics%s" % (" (cutoff %d)" % args.cutoff if args.cutoff else ""))
    print("entities: %d" % len(g.nbrs))
    print("unique triples (first occurrence): %d" % len(g.first))
    print("connected entity pairs: %d" % len(g.pair))
    print("relation types: %d" % len(g.relations))
    if years:
        print("years: %d .. %d" % (min(years), max(years)))
    print("\n## Relations (count)")
    for r, c in g.rel_count.most_common():
        print("  %-40s %d" % (r, c))
    print("\n## Top hubs (degree)")
    for n, d in sorted(g.deg.items(), key=lambda x: -x[1])[: args.top]:
        print("  %-40s %d" % (n, d))
    print("\n## New triples per year")
    for y in sorted(years):
        print("  %d %d" % (y, years[y]))


def cmd_clean(args):
    tr = read_triples(args.input)
    cfg = {}
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            cfg = json.load(fh)
    rename = cfg.get("rename", {})
    swap = cfg.get("swap", {})
    drop = set(cfg.get("drop", []))
    syn = cfg.get("synonyms", {})
    keep_only = set(cfg.get("keep_only", [])) or None
    polarity = cfg.get("polarity", [])
    norm_rel = (lambda r: re.sub(r"[\s\-]+", "_", r.strip()).upper()) if args.upper_relations else (lambda r: r.strip())

    def norm_ent(x):
        x = re.sub(r"\s+", " ", x.strip())
        x = syn.get(x, x)
        return x.lower() if args.lowercase else x

    stats = Counter()
    first = {}
    types = {}
    for s, r, o, y, extra in tr:
        s, o, r = norm_ent(s), norm_ent(o), norm_rel(r)
        if r in drop:
            stats["dropped_relation"] += 1
            continue
        if r in swap:
            s, o, r = o, s, swap[r]
            stats["swapped"] += 1
        elif r in rename:
            r = rename[r]
            stats["renamed"] += 1
        if keep_only is not None and r not in keep_only:
            stats["not_in_keep_only"] += 1
            continue
        if s == o:
            stats["self_loop"] += 1
            continue
        k = (s, r, o)
        if k in first:
            stats["duplicate"] += 1
            if y < first[k]:
                first[k] = y
        else:
            first[k] = y
        for ent, col in ((s, "subject_type"), (o, "object_type")):
            if extra.get(col) and ent not in types:
                types[ent] = extra[col]
    # polarity reconciliation (paper Table E2), per directed pair
    for grp in polarity:
        parent = grp["parent"]
        children = grp["children"]
        if len(children) != 2:
            raise SystemExit("polarity group needs exactly two children: %r" % grp)
        pos_c, neg_c = children
        pairs = set((s, o) for (s, r, o) in first if r in (pos_c, neg_c))
        for s, o in pairs:
            yp = first.get((s, pos_c, o))
            yn = first.get((s, neg_c, o))
            ypar = first.get((s, parent, o))
            if yp is not None and yn is not None:
                del first[(s, pos_c, o)]
                del first[(s, neg_c, o)]
                y = min(yp, yn) if ypar is None else min(yp, yn, ypar)
                first[(s, parent, o)] = y
                stats["polarity_conflict_resolved"] += 1
            else:
                yc = yp if yp is not None else yn
                if ypar is None:
                    first[(s, parent, o)] = yc
                    stats["parent_added"] += 1
    rc = Counter(r for (_s, r, _o) in first)
    rare = {r for r, c in rc.items() if c < args.min_relation_count}
    rows = []
    for (s, r, o), y in sorted(first.items(), key=lambda x: (x[1], x[0])):
        if r in rare:
            stats["rare_relation_removed"] += 1
            continue
        rows.append({"subject": s, "relation": r, "object": o, "year": y,
                     "subject_type": types.get(s, ""), "object_type": types.get(o, "")})
    write_rows(args.output, ["subject", "relation", "object", "year", "subject_type", "object_type"], rows)
    _log("[clean] input rows: %d -> output triples: %d" % (len(tr), len(rows)))
    for k, v in sorted(stats.items()):
        _log("[clean]   %s: %d" % (k, v))
    if rare:
        _log("[clean]   rare relations removed: %s" % ", ".join(sorted(rare)))


def _add_model_args(p):
    p.add_argument("--tau", type=float, default=10.0, help="recency decay (time units) for edge weights")
    p.add_argument("--hops", type=int, default=2, choices=[2, 3], help="max path length used by the scorer")
    p.add_argument("--nbr-cap", type=int, default=40, help="neighbours kept per node for 3-hop paths")
    p.add_argument("--mu", type=float, default=0.3, help="weight of node-profile prior in label distribution")
    p.add_argument("--hub-cap", type=int, default=1000, help="skip intermediates with degree above this when generating candidates")
    p.add_argument("--max-pairs", type=int, default=None, help="stop candidate generation after this many pairs")
    p.add_argument("--seeds", help="file with one entity per line restricting candidates")
    p.add_argument("--seed-mode", choices=["any", "both"], default="any",
                   help="any: >=1 endpoint is a seed (2-hop candidates); both: all seed-seed pairs")
    p.add_argument("--include-connected", action="store_true",
                   help="also predict new relation types for already-connected pairs")


def cmd_predict(args):
    tr = read_triples(args.triples)
    cutoff = args.cutoff if args.cutoff is not None else max(t[3] for t in tr)
    g = TKG(tr, cutoff=cutoff, tau=args.tau)
    seeds = read_list(args.seeds)
    _log("[predict] graph at %d: %d entities, %d triples" % (cutoff, len(g.nbrs), len(g.first)))
    scorer = Scorer(g, hops=args.hops, nbr_cap=args.nbr_cap, mu=args.mu)
    pairs = candidate_pairs(g, seeds, args.seed_mode, args.include_connected, args.hub_cap, args.max_pairs)
    _log("[predict] candidate pairs: %d" % len(pairs))
    params = pooled = None
    if args.calibrate:
        params, pooled = calibrate(tr, cutoff, args.calibrate, args, seeds)
    rows = []
    by_pair = defaultdict(list)
    for s, r, o, val, e in score_pairs(scorer, pairs, args.include_connected, args.min_score):
        by_pair[pkey(s, o)].append((val, s, r, o, e))
    for k, lst in by_pair.items():
        lst.sort(reverse=True)
        for val, s, r, o, e in lst[: args.labels_per_pair]:
            row = {"subject": s, "relation": r, "object": o, "score": round(val, 6),
                   "pair_support": round(e, 6)}
            if params is not None:
                p = apply_platt(params, pooled, r, val)
                row["prob"] = "" if p is None else round(p, 4)
            rows.append(row)
    rows.sort(key=lambda x: -x["score"])
    # percentile rank within relation (calibration-free comparability)
    by_rel = defaultdict(list)
    for i, row in enumerate(rows):
        by_rel[row["relation"]].append(i)
    for idxs in by_rel.values():
        n = len(idxs)
        for rank, i in enumerate(idxs):
            rows[i]["rel_percentile"] = round(1 - rank / n, 4)
    if args.min_prob is not None and params is not None:
        rows = [r for r in rows if r.get("prob") not in ("", None) and r["prob"] >= args.min_prob]
    if args.limit:
        rows = rows[: args.limit]
    fields = ["subject", "relation", "object", "score", "rel_percentile", "pair_support"]
    if params is not None:
        fields.insert(4, "prob")
    write_rows(args.output, fields, rows)
    _log("[predict] wrote %d hypotheses" % len(rows))


def _dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def cmd_backtest(args):
    tr = read_triples(args.triples)
    T = args.cutoff
    g = TKG(tr, cutoff=T, tau=args.tau)
    full = TKG(tr, cutoff=None, tau=args.tau)
    seeds = read_list(args.seeds)
    scorer = Scorer(g, hops=args.hops, nbr_cap=args.nbr_cap, mu=args.mu)
    pairs = candidate_pairs(g, seeds, args.seed_mode, args.include_connected, args.hub_cap, args.max_pairs)
    seedset = set(seeds) if seeds else None
    _log("[backtest] cutoff %d: %d entities, %d triples, %d candidate pairs"
         % (T, len(g.nbrs), len(g.first), len(pairs)))
    items = []  # (s, r, o, score)
    pair_scores = defaultdict(dict)
    for s, r, o, val, _e in score_pairs(scorer, pairs, args.include_connected):
        items.append((s, r, o, val))
        pair_scores[pkey(s, o)][(s, r, o)] = val
    # predicted set: top fraction per relation (emulates per-label thresholds)
    by_rel = defaultdict(list)
    for it in items:
        by_rel[it[1]].append(it)
    predicted = set()
    sizes = {}
    for r, lst in by_rel.items():
        lst.sort(key=lambda x: -x[3])
        k = max(1, int(math.ceil(len(lst) * args.top_frac)))
        sizes[r] = k
        predicted.update(x[:3] for x in lst[:k])
    # evaluation sets per bin
    bins = [(T + i * args.bin, T + (i + 1) * args.bin) for i in range(args.bins)]
    rng = random.Random(args.seed)
    print("# Back-test (cutoff %d, top %.1f%% per relation, %d scored hypotheses)"
          % (T, 100 * args.top_frac, len(items)))
    print("Eval set = triples first published in each bin whose entity pair was"
          " unconnected at the cutoff and whose entities both existed then.")
    print("Precision is not reported: unobserved != false (positive-unlabeled setting).\n")
    hdr = "| bin | new triples | covered | micro recall | null recall | macro recall | null macro | mean nDCG |"
    print(hdr)
    print("|" + "---|" * 8)
    results = []
    for lo, hi in bins:
        ev = []
        for (s, r, o), y in full.first.items():
            if not (lo < y <= hi):
                continue
            if s not in g.nbrs or o not in g.nbrs:
                continue
            if not args.include_connected and g.connected(s, o):
                continue
            if args.include_connected and (s, r, o) in g.first:
                continue
            if seedset is not None:
                if args.seed_mode == "both" and not (s in seedset and o in seedset):
                    continue
                if args.seed_mode == "any" and not (s in seedset or o in seedset):
                    continue
            ev.append((s, r, o))
        if not ev:
            print("| (%d, %d] | 0 | - | - | - | - | - | - |" % (lo, hi))
            continue
        evset = set(ev)
        hit = [t for t in ev if t in predicted]
        covered = sum(1 for t in ev if pkey(t[0], t[2]) in pair_scores)
        micro = len(hit) / len(ev)
        # null: random empirical scores -> random subset of same size per relation
        null_micro = []
        null_macro = []
        for _ in range(args.null_runs):
            npred = set()
            for r, lst in by_rel.items():
                npred.update(x[:3] for x in rng.sample(lst, sizes[r]))
            null_micro.append(sum(1 for t in ev if t in npred) / len(ev))
            per = defaultdict(lambda: [0, 0])
            for t in ev:
                per[t[1]][1] += 1
                per[t[1]][0] += t in npred
            vals = [h / n for h, n in per.values() if n >= args.min_support]
            null_macro.append(sum(vals) / len(vals) if vals else 0.0)
        per = defaultdict(lambda: [0, 0])
        for t in ev:
            per[t[1]][1] += 1
            per[t[1]][0] += t in predicted
        vals = [h / n for h, n in per.values() if n >= args.min_support]
        macro = sum(vals) / len(vals) if vals else float("nan")
        # label-ranking nDCG per evaluated pair
        nd = []
        ev_by_pair = defaultdict(set)
        for t in evset:
            ev_by_pair[pkey(t[0], t[2])].add(t)
        for pk, truth in ev_by_pair.items():
            ps = pair_scores.get(pk)
            if not ps:
                continue
            ranked = sorted(ps.items(), key=lambda x: -x[1])
            rels = [1.0 if tr_ in truth else 0.0 for tr_, _ in ranked]
            ideal = _dcg([1.0] * min(len(truth), len(rels)))
            if ideal > 0:
                nd.append(_dcg(rels) / ideal)
        mnd = sum(nd) / len(nd) if nd else float("nan")
        row = dict(bin="(%d, %d]" % (lo, hi), n=len(ev), covered=covered / len(ev), micro=micro,
                   null_micro=sum(null_micro) / len(null_micro), macro=macro,
                   null_macro=sum(null_macro) / len(null_macro), ndcg=mnd)
        results.append(row)
        print("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
            row["bin"], row["n"], row["covered"], row["micro"], row["null_micro"],
            row["macro"], row["null_macro"], row["ndcg"]))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"cutoff": T, "top_frac": args.top_frac, "bins": results}, fh, indent=2)
    print("\nRead: model >> null means the graph encodes predictive signal about future"
          " findings; a slow decay across bins is expected (paper: ~8% macro-recall drop"
          " over 10 years). 'covered' bounds recall: pairs never generated as candidates"
          " cannot be predicted (raise --hops / --hub-cap or use seeds).")


def _enumerate_paths(g, s, o, max_len, cap, nbr_cap):
    """Simple undirected paths s..o with 2..max_len edges (hop-limited DFS)."""
    paths = []

    def nb(x):
        ns = g.nbrs.get(x, set())
        if len(ns) > nbr_cap:
            return sorted(ns, key=lambda n: -g.weight(x, n))[:nbr_cap] + ([o] if o in ns else [])
        return ns

    stack = [(s, [s])]
    while stack and len(paths) < cap:
        x, path = stack.pop()
        if len(path) - 1 >= max_len:
            continue
        for y in nb(x):
            if y in path:
                continue
            if y == o:
                if len(path) >= 2:
                    paths.append(path + [o])
                continue
            stack.append((y, path + [y]))
    return paths


def cmd_explain(args):
    tr = read_triples(args.triples)
    cutoff = args.cutoff if args.cutoff is not None else max(t[3] for t in tr)
    g = TKG(tr, cutoff=cutoff, tau=args.tau)
    s, r, o = args.subject, args.relation, args.object
    for x in (s, o):
        if x not in g.nbrs:
            raise SystemExit("entity %r not in graph at cutoff %d" % (x, cutoff))
    hops = max(args.hops, min(3, args.max_len))
    scorer = Scorer(g, hops=hops, nbr_cap=args.nbr_cap, mu=args.mu)
    lab = r + ">"
    base_all, base_e, _ = scorer.score(s, o)
    f = base_all.get(lab, 0.0)
    rank = sorted(base_all, key=lambda k: -base_all[k])
    already = (s, r, o) in g.first
    paths = _enumerate_paths(g, s, o, args.max_len, args.max_paths, args.nbr_cap)
    cands = []
    for p in paths:
        E = set()
        for a, b in zip(p, p[1:]):
            for t in g.triples_between(a, b):
                E.add(t[:3])
        if already:
            E.discard((s, r, o))
        suff = scorer.score(s, o, only=E)[0].get(lab, 0.0)
        nec = f - scorer.score(s, o, exclude=E)[0].get(lab, 0.0)
        cands.append({"path": p, "E": E, "suff": suff, "nec": nec})
    norm = f if f > 0 else 1.0
    for c in cands:
        su, ne = c["suff"] / norm, max(0.0, c["nec"]) / norm
        if args.strategy == "sufficiency":
            c["psi"] = su
        elif args.strategy == "necessity":
            c["psi"] = ne
        else:
            c["psi"] = 2 * su * ne / (su + ne) if su + ne > 0 else 0.0
    cands.sort(key=lambda c: -c["psi"])
    pool = cands[: max(args.top * 5, args.top)]
    # diversity rerank (MMR over intermediate entities and relation signature)
    chosen = []
    while pool and len(chosen) < args.top:
        best, bval = None, -1e9
        for c in pool:
            mids = set(c["path"][1:-1])
            sig = tuple(sorted(t[1] for t in c["E"]))
            pen = 0.0
            for d in chosen:
                dm = set(d["path"][1:-1])
                jac = len(mids & dm) / max(1, len(mids | dm))
                same_sig = sig == tuple(sorted(t[1] for t in d["E"]))
                pen = max(pen, jac + (0.3 if same_sig else 0.0))
            val = c["psi"] - args.diversity * pen
            if val > bval:
                best, bval = c, val
        chosen.append(best)
        pool.remove(best)
    latent = []
    if args.latent:
        half = sorted(g.nbrs[s] ^ g.nbrs[o],
                      key=lambda n: -(g.weight(s, n) + g.weight(n, o)))[:500]
        for n in half:
            if n in (s, o):
                continue
            a, b = (s, n) if n in g.nbrs[s] else (n, o)
            gap = (n, o) if a == s else (s, n)
            sc = scorer.score(*gap)[0]
            if not sc:
                continue
            tok, val = max(sc.items(), key=lambda x: x[1])
            latent.append({"observed": [t[:3] + (t[3],) for t in g.triples_between(a, b)],
                           "predicted": tok_to_triple(gap[0], gap[1], tok), "score": val})
        latent.sort(key=lambda x: -x["score"])
        latent = latent[: args.top]

    def fmt(t):
        yr = g.first.get(t)
        return "(%s, %s, %s)%s" % (t[0], t[1], t[2], " [%d]" % yr if yr else "")

    out = {
        "hypothesis": [s, r, o], "cutoff": cutoff, "score": f, "pair_support": base_e,
        "label_rank": rank.index(lab) + 1 if lab in rank else None, "n_labels": len(rank),
        "already_known": already, "shortest_path": g.shortest_path_len(s, o),
        "explanations": [
            {"path": c["path"], "facts": [list(t) + [g.first.get(t)] for t in sorted(c["E"])],
             "sufficiency": c["suff"], "necessity": c["nec"], "influence": c["psi"]}
            for c in chosen],
        "latent_paths": [{"observed": [list(x) for x in l["observed"]], "predicted": list(l["predicted"]),
                          "score": l["score"]} for l in latent],
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
    print("# Explanation for (%s, %s, %s) at cutoff %d" % (s, r, o, cutoff))
    if already:
        print("> NOTE: this triple is already in the graph (first seen %d) - it is not novel."
              % g.first[(s, r, o)])
    print("score=%.4f  pair_support=%.4f  label rank %s/%d  shortest path=%s  paths examined=%d"
          "  (scorer hops=%d)\n"
          % (f, base_e, out["label_rank"], len(rank), out["shortest_path"], len(cands), hops))
    if not chosen:
        print("No connecting paths within %d hops: the prediction has no structural support"
              " in this graph (treat as speculative)." % args.max_len)
    for i, c in enumerate(chosen, 1):
        print("## E%d  influence=%.3f  sufficiency=%.4f  necessity=%.4f" % (i, c["psi"], c["suff"], c["nec"]))
        print("   path: " + " -- ".join(c["path"]))
        for t in sorted(c["E"], key=lambda t: (c["path"].index(t[0]) if t[0] in c["path"] else 0)):
            print("   - " + fmt(t))
        print()
    if latent:
        print("## Latent paths (one leg observed, one leg predicted)")
        for l in latent:
            obs = "; ".join("(%s, %s, %s) [%d]" % tuple(x) for x in l["observed"])
            print("   - %s  +  predicted (%s, %s, %s) score=%.4f" % ((obs,) + tuple(l["predicted"]) + (l["score"],)))


def cmd_select(args):
    tr = read_triples(args.triples)
    cutoff = args.cutoff if args.cutoff is not None else max(t[3] for t in tr)
    g = TKG(tr, cutoff=cutoff, tau=args.tau)
    with open(args.predictions, newline="", encoding="utf-8") as fh:
        preds = list(csv.DictReader(fh))
    col = args.score_col
    if preds and col not in preds[0]:
        raise SystemExit("score column %r not in predictions (have %s)" % (col, list(preds[0])))
    fields = [c for c in preds[0].keys() if not c.startswith("_")] if preds else []
    fields += ["recency_subject", "recency_object", "shortest_path"]
    log = [("input", len(preds), "raw model output")]
    seen = set()
    stage = []
    for p in preds:
        k = (p["subject"], p["relation"], p["object"])
        if k in seen:
            continue
        seen.add(k)
        if args.allow_connected:
            if k in g.first:
                continue
        elif g.connected(p["subject"], p["object"]):
            continue
        try:
            p["_score"] = float(p[col])
        except (TypeError, ValueError):
            continue
        stage.append(p)
    log.append(("prepare", len(stage), "dedupe + remove relations already in graph"))
    if args.min_score is not None:
        stage = [p for p in stage if p["_score"] >= args.min_score]
        log.append(("threshold", len(stage), "%s >= %s" % (col, args.min_score)))
    exclude = set(read_list(args.exclude_entities) or [])
    if exclude:
        stage = [p for p in stage if p["subject"] not in exclude and p["object"] not in exclude]
        log.append(("exclude", len(stage), "removed %d excluded entities" % len(exclude)))
    stage.sort(key=lambda p: -p["_score"])
    keep = set()
    per_ent = defaultdict(int)
    for i, p in enumerate(stage):
        for ent in (p["subject"], p["object"]):
            if per_ent[ent] < args.top_k:
                per_ent[ent] += 1
                keep.add(i)
    stage = [p for i, p in enumerate(stage) if i in keep]
    log.append(("top-k per entity", len(stage), "k=%d (union over subject/object)" % args.top_k))
    for p in stage:
        p["recency_subject"] = g.recency(p["subject"])
        p["recency_object"] = g.recency(p["object"])
        p["shortest_path"] = g.shortest_path_len(p["subject"], p["object"], args.max_path)
    if args.min_path:
        stage = [p for p in stage if p["shortest_path"] is None or p["shortest_path"] >= args.min_path]
        log.append(("min path", len(stage), "shortest path >= %d (less obvious hypotheses)" % args.min_path))
    if args.limit:
        stage = stage[: args.limit]
        log.append(("limit", len(stage), "top %d by score" % args.limit))
    write_rows(args.output, fields, stage)
    rep = ["# Hypothesis selection report", "", "| step | output | note |", "|---|---|---|"]
    rep += ["| %s | %d | %s |" % x for x in log]
    pl = Counter(p["shortest_path"] for p in stage)
    rep += ["", "## Shortest-path length distribution", "", "| length | count |", "|---|---|"]
    rep += ["| %s | %d |" % (k if k is not None else ">%d/none" % args.max_path, v)
            for k, v in sorted(pl.items(), key=lambda x: (x[0] is None, x[0] or 0))]
    text = "\n".join(rep) + "\n"
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(text)
    _log(text)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="hakken_kg.py",
        description="Hakken-style knowledge prediction on temporal knowledge graphs (stdlib only).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("stats", help="inspect a triple file")
    p.add_argument("triples")
    p.add_argument("--cutoff", type=int)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--tau", type=float, default=10.0)
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("clean", help="normalise and reconcile a triple file")
    p.add_argument("input")
    p.add_argument("-o", "--output", default="-")
    p.add_argument("--config", help="JSON with rename/swap/drop/keep_only/synonyms/polarity")
    p.add_argument("--min-relation-count", type=int, default=1)
    p.add_argument("--lowercase", action="store_true", help="casefold entity names")
    p.add_argument("--upper-relations", action="store_true", help="UPPER_SNAKE relation names")
    p.set_defaults(fn=cmd_clean)

    p = sub.add_parser("predict", help="score novel hypotheses")
    p.add_argument("triples")
    p.add_argument("-o", "--output", default="-")
    p.add_argument("--cutoff", type=int, help="knowledge cutoff (default: latest year)")
    p.add_argument("--calibrate", type=int, metavar="W",
                   help="fit Platt calibration on the window (cutoff-W, cutoff]")
    p.add_argument("--labels-per-pair", type=int, default=3)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--min-prob", type=float)
    p.add_argument("--limit", type=int)
    _add_model_args(p)
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("backtest", help="time-sliced evaluation vs. null baseline")
    p.add_argument("triples")
    p.add_argument("--cutoff", type=int, required=True)
    p.add_argument("--bin", type=int, default=2, help="bin width (time units)")
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--top-frac", type=float, default=0.1, help="fraction of scored hypotheses per relation counted as predicted")
    p.add_argument("--null-runs", type=int, default=5)
    p.add_argument("--min-support", type=int, default=3, help="min eval triples for a relation to enter macro recall")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json")
    _add_model_args(p)
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("explain", help="PHELInE-style path explanations for one hypothesis")
    p.add_argument("triples")
    p.add_argument("--subject", "-s", required=True)
    p.add_argument("--relation", "-r", required=True)
    p.add_argument("--object", "-O", required=True)
    p.add_argument("--cutoff", type=int)
    p.add_argument("--max-len", type=int, default=3, help="max path length (edges)")
    p.add_argument("--max-paths", type=int, default=3000)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--strategy", choices=["both", "sufficiency", "necessity"], default="both")
    p.add_argument("--diversity", type=float, default=0.5)
    p.add_argument("--latent", action="store_true", help="also report half-observed, half-predicted paths")
    p.add_argument("--json")
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--hops", type=int, default=2, choices=[2, 3])
    p.add_argument("--nbr-cap", type=int, default=40)
    p.add_argument("--mu", type=float, default=0.3)
    p.set_defaults(fn=cmd_explain)

    p = sub.add_parser("select", help="hypothesis-selection pipeline")
    p.add_argument("predictions")
    p.add_argument("--triples", required=True)
    p.add_argument("--cutoff", type=int)
    p.add_argument("-o", "--output", default="-")
    p.add_argument("--report")
    p.add_argument("--score-col", default="score")
    p.add_argument("--min-score", type=float)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--min-path", type=int, default=0)
    p.add_argument("--max-path", type=int, default=6)
    p.add_argument("--limit", type=int)
    p.add_argument("--exclude-entities", help="file of entities to drop (e.g. overly generic hubs)")
    p.add_argument("--allow-connected", action="store_true")
    p.add_argument("--tau", type=float, default=10.0)
    p.set_defaults(fn=cmd_select)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
