"""Temporal knowledge graph: config application (cleaning), snapshots, pattern mining, candidates."""
from __future__ import annotations

import math
import random
import re
import statistics
from collections import Counter, defaultdict

from .common import die, log, pkey

# --------------------------------------------------------------------------- #
# Cleaning (Hakken App. E): normalisation, relation merging, polarity, first occurrence
# --------------------------------------------------------------------------- #


def apply_config(raw, rel_cfg=None, ent_cfg=None, verbose=True):
    """raw: iterable of (s, r, o, year, extra). Returns (first, support, types, stats).

    first[(s, r, o)]   -> earliest year
    support[(s, r, o)] -> number of distinct sources observing the triple
    """
    rel_cfg = rel_cfg or {}
    ent_cfg = ent_cfg or {}
    rename = rel_cfg.get("rename", {})
    swap = rel_cfg.get("swap", {})
    drop = set(rel_cfg.get("drop", []))
    keep_only = set(rel_cfg.get("keep_only", [])) or None
    polarity = rel_cfg.get("polarity", [])
    upper = rel_cfg.get("upper_snake", True)
    syn = ent_cfg.get("synonyms", {})
    lower = ent_cfg.get("lowercase", False)

    def nrel(r):
        r = r.strip()
        return re.sub(r"[\s\-]+", "_", r).upper() if upper else r

    def nent(x):
        x = re.sub(r"\s+", " ", x.strip())
        x = syn.get(x, x)
        return x.lower() if lower else x

    stats = Counter()
    first, sources, types = {}, defaultdict(set), {}
    for s, r, o, y, extra in raw:
        s, o, r = nent(s), nent(o), nrel(r)
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
            stats["repeat_observation"] += 1
            if y < first[k]:
                first[k] = y
        else:
            first[k] = y
        src = (extra or {}).get("source") or "row%d" % stats["rows"]
        sources[k].add(src)
        stats["rows"] += 1
        for ent, col in ((s, "subject_type"), (o, "object_type")):
            t = (extra or {}).get(col)
            if t and ent not in types:
                types[ent] = t
    for grp in polarity:
        parent, children = grp["parent"], grp["children"]
        if len(children) != 2:
            die("polarity group needs exactly two children: %r" % grp)
        pos_c, neg_c = children
        for s, o in {(s, o) for (s, r, o) in first if r in (pos_c, neg_c)}:
            yp, yn = first.get((s, pos_c, o)), first.get((s, neg_c, o))
            ypar = first.get((s, parent, o))
            if yp is not None and yn is not None:  # contradictory children -> parent only
                srcs = sources.pop((s, pos_c, o), set()) | sources.pop((s, neg_c, o), set())
                del first[(s, pos_c, o)]
                del first[(s, neg_c, o)]
                first[(s, parent, o)] = min(x for x in (yp, yn, ypar) if x is not None)
                sources[(s, parent, o)] |= srcs
                stats["polarity_conflict_resolved"] += 1
            elif ypar is None:  # child only -> add the more general parent
                yc = yp if yp is not None else yn
                first[(s, parent, o)] = yc
                sources[(s, parent, o)] |= sources[(s, pos_c if yp is not None else neg_c, o)]
                stats["parent_added"] += 1
    minc = int(rel_cfg.get("min_relation_count", 1) or 1)
    if minc > 1:
        rc = Counter(r for (_s, r, _o) in first)
        rare = {r for r, c in rc.items() if c < minc}
        for k in [k for k in first if k[1] in rare]:
            del first[k]
            stats["rare_relation_removed"] += 1
    support = {k: len(sources.get(k, ())) or 1 for k in first}
    if verbose and stats:
        changes = ", ".join("%s=%d" % kv for kv in sorted(stats.items()) if kv[0] != "rows")
        log("[graph] %d observations -> %d unique triples%s" % (
            stats["rows"], len(first), " (%s)" % changes if changes else ""))
    return first, support, types, stats


# --------------------------------------------------------------------------- #
# Snapshot
# --------------------------------------------------------------------------- #


class TKG:
    """Snapshot of the literature graph up to (and including) `cutoff`.

    Edge weight = sum over the pair's triples of 0.5 + 0.5*exp(-(T - year)/tau):
    evidence accumulates, recent evidence counts more, old evidence never vanishes.
    """

    def __init__(self, first, cutoff=None, tau=10.0, support=None, types=None):
        self.first = {k: y for k, y in first.items() if cutoff is None or y <= cutoff}
        self.T = cutoff if cutoff is not None else max(self.first.values(), default=0)
        self.tau = float(tau)
        self.support = support or {}
        self.types = types or {}
        self.pair = defaultdict(list)
        self.nbrs = defaultdict(set)
        self.out_rel = defaultdict(Counter)
        self.in_rel = defaultdict(Counter)
        self.rel_count = Counter()
        self.node_years = defaultdict(list)
        self.edge_year = {}
        for (s, r, o), y in self.first.items():
            k = pkey(s, o)
            self.pair[k].append((s, r, o, y))
            self.edge_year[k] = min(y, self.edge_year.get(k, y))
            self.nbrs[s].add(o)
            self.nbrs[o].add(s)
            self.out_rel[s][r] += 1
            self.in_rel[o][r] += 1
            self.rel_count[r] += 1
            self.node_years[s].append(y)
            self.node_years[o].append(y)
        self.relations = sorted(self.rel_count)
        self.deg = {n: len(v) for n, v in self.nbrs.items()}
        self.node_first = {n: min(ys) for n, ys in self.node_years.items()}
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

    def first_year(self, a, b, exclude=None, only=None):
        ys = [t[3] for t in self.pair.get(pkey(a, b), ())
              if (exclude is None or t[:3] not in exclude) and (only is None or t[:3] in only)]
        return min(ys) if ys else None

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
        return list(self.pair.get(pkey(a, b), ()))

    def recency(self, n):
        ys = self.node_years.get(n)
        if not ys:
            return None
        m = statistics.median(ys)
        return int(m) if m == int(m) else round(m, 1)

    def growth(self, n, window=3):
        """Fraction of a node's neighbours first connected within the last `window` units."""
        ns = self.nbrs.get(n)
        if not ns:
            return 0.0
        recent = sum(1 for m in ns if self.edge_year[pkey(n, m)] > self.T - window)
        return recent / len(ns)

    def shortest_path_len(self, a, b, max_len=6):
        if a not in self.nbrs or b not in self.nbrs:
            return None
        if a == b:
            return 0
        seen, frontier = {a}, [a]
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


def mine_patterns(g, max_common=100, seed=13):
    """conf(l | t1, t2) ~= P(label l on u-v | u -t1- n -t2- v and u-v connected)."""
    rng = random.Random(seed)
    closed, total = Counter(), Counter()
    for (a, b), trips in g.pair.items():
        common = g.nbrs[a] & g.nbrs[b]
        if not common:
            continue
        if len(common) > max_common:
            common = rng.sample(sorted(common), max_common)
        for u, v in ((a, b), (b, a)):
            labs = {r + (">" if s == u else "<") for s, r, o, _ in trips}
            for n in common:
                for t1 in g.tokens(u, n):
                    for t2 in g.tokens(n, v):
                        total[(t1, t2)] += 1
                        for lab in labs:
                            closed[(t1, t2, lab)] += 1
    return closed, total


class LabelModel:
    """Relation-type side shared by all scorers: pattern rules + node profiles + priors."""

    def __init__(self, g, beta=2.0, max_common=100):
        self.g = g
        closed, total = mine_patterns(g, max_common=max_common)
        cnt = Counter()
        for (s, r, o) in g.first:
            cnt[r + ">"] += 1
            cnt[r + "<"] += 1
        z = sum(cnt.values()) or 1
        self.prior = {k: v / z for k, v in cnt.items()}
        self.labels = sorted(self.prior)
        by_pat = defaultdict(dict)
        for (t1, t2, lab), c in closed.items():
            by_pat[(t1, t2)][lab] = c
        self.conf = {}
        for pat, tot in total.items():
            d = by_pat.get(pat, {})
            self.conf[pat] = {lab: (d.get(lab, 0) + beta * p) / (tot + beta)
                              for lab, p in self.prior.items() if d.get(lab, 0) > 0 or p > 0.01}

    def pattern_dist(self, t1s, t2s):
        acc, n = Counter(), 0
        for t1 in t1s:
            for t2 in t2s:
                for lab, p in self.conf.get((t1, t2), self.prior).items():
                    acc[lab] += p
                n += 1
        return {k: v / n for k, v in acc.items()} if n else {}

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
        return {k: v / z for k, v in d.items()} if z > 0 else {}


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #


def candidate_pairs(g, seeds=None, seed_mode="any", include_connected=False,
                    hub_cap=1000, max_pairs=None):
    pairs = set()
    if seeds is not None:
        missing = [s for s in seeds if s not in g.nbrs]
        if missing:
            log("[candidates] %d seed(s) absent from the graph at %s: %s%s" % (
                len(missing), g.T, ", ".join(missing[:5]), " ..." if len(missing) > 5 else ""))
        seeds = [s for s in seeds if s in g.nbrs]
    if seeds is not None and seed_mode == "both":
        ss = sorted(set(seeds))
        for i, a in enumerate(ss):
            for b in ss[i + 1:]:
                if include_connected or not g.connected(a, b):
                    pairs.add((a, b))
    else:
        sources = seeds if seeds is not None else sorted(g.nbrs)
        for u in sources:
            for n in g.nbrs[u]:
                if g.deg[n] > hub_cap:
                    continue
                for v in g.nbrs[n]:
                    if v == u or (not include_connected and g.connected(u, v)):
                        continue
                    pairs.add(pkey(u, v))
                    if max_pairs and len(pairs) >= max_pairs:
                        log("[candidates] stopped at --max-pairs %d" % max_pairs)
                        return sorted(pairs)
    return sorted(pairs)
