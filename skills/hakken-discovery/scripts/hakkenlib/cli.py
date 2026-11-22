"""Command-line interface. Run `python3 scripts/hakken.py -h`."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

from . import __version__
from .common import (die, hyp_key, log, logit, now_iso, pkey, read_jsonl, read_list,
                     read_json, read_table, read_triples, sigmoid, tok_to_triple, write_json,
                     write_rows, write_text)
from .graph import TKG, apply_config, candidate_pairs
from .models import (FeatureScorer, HeuristicScorer, apply_platt, calibrate_heuristic,
                     describe_weights, fit_feature_model)
from .workspace import Workspace, find_workspace, init_workspace

PRED_FIELDS = ["subject", "relation", "object", "score", "prob", "rel_percentile", "support", "model"]
PROMPT_VERSION = "judge-v1"

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def get_ws(args, required=True):
    root = find_workspace(getattr(args, "workspace", None))
    if root is None:
        if required:
            die("no hakken workspace found (looked for hakken/hakken.json from %s upward). "
                "Create one with: hakken.py init" % os.getcwd())
        return None
    return Workspace(root)


def load_first(args, ws):
    """Return (first, support, types, rel_cfg, ent_cfg) from --triples or the workspace ledger."""
    rel_cfg = ws.relations_cfg() if ws else {}
    ent_cfg = ws.entities_cfg() if ws else {}
    if getattr(args, "config", None):
        cfg = read_json(args.config, {})
        rel_cfg = dict(rel_cfg, **{k: v for k, v in cfg.items() if k not in ("synonyms", "exclude", "lowercase")})
        ent_cfg = dict(ent_cfg, **{k: v for k, v in cfg.items() if k in ("synonyms", "exclude", "lowercase")})
    if getattr(args, "triples", None):
        raw = read_triples(args.triples)
    elif ws is not None:
        raw = ws.ledger_triples()
        if not raw:
            die("the ledger is empty - add data with `ingest` or `fetch-pubtator`")
    else:
        die("give --triples FILE or run inside a workspace")
    first, support, types, _ = apply_config(raw, rel_cfg, ent_cfg)
    return first, support, types, rel_cfg, ent_cfg


def resolve_seeds(args, ws):
    if getattr(args, "seeds", None):
        return read_list(args.seeds)
    if ws is not None and not getattr(args, "no_seeds", False):
        return ws.seeds()
    return None


def check_external(ws, policy_key, confirmed, what):
    """Enforce workspace policy for anything that sends data to third parties."""
    pol = ws.policy(policy_key, "ask") if ws else "ask"
    if pol == "deny":
        die("%s is disabled by policy %s=deny in hakken.json" % (what, policy_key))
    if pol == "ask" and not confirmed:
        die("%s sends queries to external services and policy %s=ask. Get the user's approval "
            "(or, in a non-interactive/autonomous mode, defer it), then re-run with --confirmed."
            % (what, policy_key), code=3)


def model_args(p, window_default=5):
    p.add_argument("--model", choices=["features", "heuristic"], default="features",
                   help="graph branch: fitted temporal features (default) or fixed heuristic")
    p.add_argument("--window", type=int, default=window_default,
                   help="history window W used to fit the features model / calibrate the heuristic "
                        "(probability = reported within the next W time units)")
    p.add_argument("--tau", type=float, default=10.0, help="recency decay of edge weights (time units)")
    p.add_argument("--hops", type=int, default=2, choices=[2, 3], help="max path length in features/scores")
    p.add_argument("--nbr-cap", type=int, default=40, help="neighbours per node considered for 3-hop paths")
    p.add_argument("--hub-cap", type=int, default=1000, help="skip intermediates with degree above this for candidates")
    p.add_argument("--max-pairs", type=int, help="cap on candidate pairs")
    p.add_argument("--max-train-pairs", type=int, default=20000)
    p.add_argument("--max-rows", type=int, default=30000, help="cap on training rows for the features model")
    p.add_argument("--seeds", help="file with one focus entity per line (default: config/seeds.txt)")
    p.add_argument("--no-seeds", action="store_true", help="ignore config/seeds.txt")
    p.add_argument("--seed-mode", choices=["any", "both"], default="any")
    p.add_argument("--include-connected", action="store_true",
                   help="also predict new relation types for already-connected pairs")


def build_scorer(args, first, support, types, cutoff, seeds):
    g = TKG(first, cutoff=cutoff, tau=args.tau, support=support, types=types)
    meta = {}
    if args.model == "features":
        params = fit_feature_model(first, cutoff, args.window, support=support, types=types, tau=args.tau,
                                   hops=args.hops, nbr_cap=args.nbr_cap, seeds=seeds, seed_mode=args.seed_mode,
                                   hub_cap=args.hub_cap, max_train_pairs=args.max_train_pairs,
                                   max_rows=args.max_rows)
        sc = FeatureScorer(g, params)
        meta = {"model": "features", "params": params}
    else:
        sc = HeuristicScorer(g, hops=args.hops, nbr_cap=args.nbr_cap)
        cal = None
        if args.window and args.window > 0:
            try:
                cal = calibrate_heuristic(first, cutoff, args.window, tau=args.tau, hops=args.hops,
                                          nbr_cap=args.nbr_cap, seeds=seeds, seed_mode=args.seed_mode,
                                          hub_cap=args.hub_cap, max_pairs=args.max_pairs, support=support)
            except Exception as e:  # calibration is optional
                log("[calibrate] skipped: %s" % e)
        sc.calibration = cal
        meta = {"model": "heuristic", "calibration": cal}
    return g, sc, meta


def scorer_from_run(ws, run_dir, args_like, first, support, types):
    """Rebuild the run's graph-branch scorer (features params are reused, not refitted)."""
    man = read_json(os.path.join(run_dir, "manifest.json"))
    p = man["params"]
    cutoff = p["cutoff"]
    g = TKG(first, cutoff=cutoff, tau=p.get("tau", 10.0), support=support, types=types)
    model = read_json(os.path.join(run_dir, "model.json"), {}) or {}
    if model.get("model") == "features":
        return g, FeatureScorer(g, model["params"]), cutoff
    note = "" if model.get("model") == "heuristic" else " (external scores: heuristic used as explanation proxy)"
    log("[explain] using heuristic scorer%s" % note)
    return g, HeuristicScorer(g, hops=p.get("hops", 2), nbr_cap=p.get("nbr_cap", 40)), cutoff


def score_candidates(sc, pairs, include_connected=False, labels_per_pair=None):
    g = sc.g
    for a, b in pairs:
        scores, sup = sc.score(a, b)
        if not scores:
            continue
        items = sorted(scores.items(), key=lambda x: -x[1])
        if labels_per_pair:
            items = items[:labels_per_pair]
        for tok, val in items:
            s, r, o = tok_to_triple(a, b, tok)
            if include_connected and (s, r, o) in g.first:
                continue
            yield s, r, o, val, sup


def _fmt(x, nd=4):
    return "" if x is None or x == "" else (round(x, nd) if isinstance(x, float) else x)


# --------------------------------------------------------------------------- #
# workspace & data commands
# --------------------------------------------------------------------------- #


def cmd_init(args):
    path = args.path or os.path.join(os.getcwd(), "hakken")
    init_workspace(path, goal=args.goal or "", domain=args.domain or "", time_unit=args.time_unit,
                   git_policy=args.git_policy, external_novelty_search=args.novelty_search,
                   entity_evidence_search=args.evidence_search, force=args.force)


def cmd_status(args):
    ws = get_ws(args)
    m = ws.manifest
    led = ws.read_ledger()
    print("workspace: %s" % ws.root)
    print("goal: %s | domain: %s | time unit: %s" % (m.get("goal") or "-", m.get("domain") or "-", m.get("time_unit")))
    print("git policy: %s | policy: %s" % (m.get("git_policy"), json.dumps(m.get("policy"))))
    print("ledger: %d observations; by extractor: %s" % (
        len(led), dict(Counter(r.get("extractor", "") for r in led).most_common())))
    if led:
        ys = [int(r["year"]) for r in led if str(r.get("year", "")).lstrip("-").isdigit()]
        if ys:
            print("years: %d .. %d" % (min(ys), max(ys)))
    seeds = ws.seeds()
    print("seeds: %s" % (len(seeds) if seeds else 0))
    print("judgments: %d | novelty checks: %d" % (len(ws.judgments()), len(ws.novelty_checks())))
    runs = ws.runs()
    print("runs (%d):" % len(runs))
    for r in runs[-8:]:
        man = read_json(ws.p("runs", r, "manifest.json"), {})
        files = sorted(f for f in os.listdir(ws.p("runs", r)) if not f.startswith("."))
        print("  %s  [%s]  %s" % (r, man.get("kind"), ", ".join(files)))


def cmd_ingest(args):
    ws = get_ws(args)
    rows = []
    for s, r, o, y, extra in read_triples(args.file):
        rows.append({"subject": s, "relation": r, "object": o, "year": y,
                     "source": extra.get(args.source_col, "") if args.source_col else extra.get("source", ""),
                     "evidence": extra.get("evidence", ""), "subject_type": extra.get("subject_type", ""),
                     "object_type": extra.get("object_type", ""), "extractor": extra.get("extractor", "")})
    added = ws.append_ledger(rows, args.extractor)
    log("[ingest] %d rows read, %d new observations appended to %s" % (len(rows), added, ws.ledger_path))


def cmd_fetch_pubtator(args):
    from .sources import pubtator_relations, pubtator_search
    ws = get_ws(args)
    check_external(ws, "entity_evidence_search", args.confirmed, "fetch-pubtator")
    pmids = []
    for q in args.query:
        if args.year_from is not None:
            # stratified by year: relevance-ranked search otherwise returns mostly recent papers,
            # which leaves no history for temporal modelling or back-testing
            got = []
            for y in range(args.year_from, (args.year_to or args.year_from) + 1):
                got += pubtator_search("%s AND year:%d" % (q, y), args.per_year)
        else:
            got = pubtator_search(q, args.max_docs)
        log("[pubtator] %r -> %d documents" % (q, len(got)))
        pmids += got
    pmids = list(dict.fromkeys(pmids))
    rows = list(pubtator_relations(pmids))
    added = ws.append_ledger(rows, "pubtator3")
    log("[pubtator] %d relation observations from %d documents; %d new in ledger" % (len(rows), len(pmids), added))


def cmd_evidence(args):
    from .sources import mention_sentences, openalex_search, pubmed_abstracts, pubmed_search
    ws = get_ws(args, required=False)
    check_external(ws, "entity_evidence_search", args.confirmed or ws is None, "evidence lookup")
    if args.source == "pubmed":
        term = '"%s"' % args.entity + (" AND (%s)" % args.context if args.context else "")
        count, ids = pubmed_search(term, before=args.before, retmax=args.docs)
        docs = pubmed_abstracts(ids)
    else:
        q = '"%s"' % args.entity + (" " + args.context if args.context else "")
        count, docs = openalex_search(q, before=args.before, per_page=args.docs)
    docs = [d for d in docs if args.before is None or (d.get("year") or 0) <= args.before]
    sents = mention_sentences(docs, args.entity, args.exclude_term or [], args.max)
    out = {"entity": args.entity, "before": args.before, "source": args.source,
           "documents_matching": count, "documents_used": len(docs), "sentences": sents}
    print(json.dumps(out, ensure_ascii=False, indent=1))


def cmd_stats(args):
    ws = get_ws(args, required=False)
    first, support, types, _, _ = load_first(args, ws)
    g = TKG(first, cutoff=args.cutoff, support=support, types=types)
    years = Counter(g.first.values())
    print("# Graph statistics%s" % (" (cutoff %s)" % args.cutoff if args.cutoff else ""))
    print("entities: %d | unique triples: %d | connected pairs: %d | relation types: %d" % (
        len(g.nbrs), len(g.first), len(g.pair), len(g.relations)))
    if years:
        print("time range: %d .. %d" % (min(years), max(years)))
    multi = sum(1 for k in g.first if support.get(k, 1) > 1)
    print("triples observed in >1 source: %d" % multi)
    print("\n## Relations")
    for r, c in g.rel_count.most_common():
        print("  %-42s %d" % (r, c))
    print("\n## Top hubs (degree) - consider merging or excluding generic ones")
    for n, d in sorted(g.deg.items(), key=lambda x: -x[1])[: args.top]:
        print("  %-42s %d" % (n, d))
    print("\n## New triples per time unit")
    for y in sorted(years):
        print("  %s %d" % (y, years[y]))


def cmd_export_graph(args):
    ws = get_ws(args, required=False)
    first, support, types, _, _ = load_first(args, ws)
    rows = [{"subject": s, "relation": r, "object": o, "year": y, "n_sources": support.get((s, r, o), 1),
             "subject_type": types.get(s, ""), "object_type": types.get(o, "")}
            for (s, r, o), y in sorted(first.items(), key=lambda x: (x[1], x[0]))
            if args.cutoff is None or y <= args.cutoff]
    write_rows(args.output, ["subject", "relation", "object", "year", "n_sources", "subject_type", "object_type"], rows)
    log("[export] %d triples" % len(rows))


# --------------------------------------------------------------------------- #
# graph branch: backtest & predict
# --------------------------------------------------------------------------- #


def _dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def _backtest_one(args, first, support, types, seeds, model):
    a = argparse.Namespace(**vars(args))
    a.model = model
    T = args.cutoff
    g, sc, meta = build_scorer(a, first, support, types, T, seeds)
    pairs = candidate_pairs(g, seeds, args.seed_mode, args.include_connected, args.hub_cap, args.max_pairs)
    items, pair_scores = [], defaultdict(dict)
    for s, r, o, val, _sup in score_candidates(sc, pairs, args.include_connected):
        if model == "heuristic" and getattr(sc, "calibration", None):
            p = apply_platt(sc.calibration, r, val)
            val = p if p is not None else val
        items.append((s, r, o, val))
        pair_scores[pkey(s, o)][(s, r, o)] = val
    by_rel = defaultdict(list)
    for it in items:
        by_rel[it[1]].append(it)
    predicted, sizes = set(), {}
    for r, lst in by_rel.items():
        lst.sort(key=lambda x: -x[3])
        k = max(1, int(math.ceil(len(lst) * args.top_frac)))
        sizes[r] = k
        predicted.update(x[:3] for x in lst[:k])
    seedset = set(seeds) if seeds else None
    rng = random.Random(args.seed)
    bins = [(T + i * args.bin, T + (i + 1) * args.bin) for i in range(args.bins)]
    rows = []
    for lo, hi in bins:
        ev = []
        for (s, r, o), y in first.items():
            if not (lo < y <= hi) or s not in g.nbrs or o not in g.nbrs:
                continue
            if args.include_connected:
                if (s, r, o) in g.first:
                    continue
            elif g.connected(s, o):
                continue
            if seedset is not None:
                if args.seed_mode == "both" and not (s in seedset and o in seedset):
                    continue
                if args.seed_mode == "any" and not (s in seedset or o in seedset):
                    continue
            ev.append((s, r, o))
        if not ev:
            rows.append({"bin": "(%s, %s]" % (lo, hi), "n": 0})
            continue

        def recalls(pred):
            per = defaultdict(lambda: [0, 0])
            for t in ev:
                per[t[1]][1] += 1
                per[t[1]][0] += t in pred
            vals = [h / n for h, n in per.values() if n >= args.min_support]
            return sum(1 for t in ev if t in pred) / len(ev), (sum(vals) / len(vals) if vals else float("nan"))

        micro, macro = recalls(predicted)
        nm, nM = [], []
        for _ in range(args.null_runs):
            npred = set()
            for r, lst in by_rel.items():
                npred.update(x[:3] for x in rng.sample(lst, sizes[r]))
            a_, b_ = recalls(npred)
            nm.append(a_)
            nM.append(b_)
        evp = defaultdict(set)
        for t in ev:
            evp[pkey(t[0], t[2])].add(t)
        nd = []
        for pk, truth in evp.items():
            ps = pair_scores.get(pk)
            if ps:
                ranked = sorted(ps.items(), key=lambda x: -x[1])
                rel = [1.0 if tr in truth else 0.0 for tr, _ in ranked]
                ideal = _dcg([1.0] * min(len(truth), len(rel)))
                if ideal > 0:
                    nd.append(_dcg(rel) / ideal)
        covered = sum(1 for t in ev if pkey(t[0], t[2]) in pair_scores) / len(ev)
        rows.append({"bin": "(%s, %s]" % (lo, hi), "n": len(ev), "covered": covered, "micro": micro,
                     "null_micro": sum(nm) / len(nm), "macro": macro,
                     "null_macro": sum(x for x in nM if x == x) / max(1, sum(1 for x in nM if x == x)),
                     "ndcg": sum(nd) / len(nd) if nd else float("nan")})
    calib = None
    if model == "features" or (model == "heuristic" and getattr(sc, "calibration", None)):
        W = args.window
        appeared = sum(1 for s, r, o, _ in items if T < first.get((s, r, o), -10**9) <= T + W)
        expected = sum(v for *_x, v in items)
        calib = {"window": W, "expected": expected, "observed": appeared, "n_items": len(items)}
    weights = describe_weights(meta["params"])[:6] if model == "features" else None
    return {"model": model, "n_pairs": len(pairs), "n_items": len(items), "bins": rows,
            "calibration_in_the_large": calib, "top_weights": weights}


def cmd_backtest(args):
    ws = get_ws(args)
    first, support, types, _, _ = load_first(args, ws)
    seeds = resolve_seeds(args, ws)
    models = ["features", "heuristic"] if args.model == "both" else [args.model]
    rd = ws.new_run("backtest", "c%s_%s" % (args.cutoff, args.model),
                    {k: v for k, v in vars(args).items() if k != "fn"}, sys.argv[1:])
    res = {"cutoff": args.cutoff, "top_frac": args.top_frac, "results": []}
    md = ["# Back-test at cutoff %s" % args.cutoff, "",
          "Eval set per bin: triples first published in the bin whose entity pair was unconnected at "
          "the cutoff and whose entities both existed then. Prediction budget: top %.0f%% of scored "
          "hypotheses per relation. Precision is not reported (unobserved != false)." % (100 * args.top_frac), ""]
    for m in models:
        try:
            r = _backtest_one(args, first, support, types, seeds, m)
        except ValueError as e:
            die("backtest (%s) failed: %s" % (m, e))
        res["results"].append(r)
        md += ["## Model: %s  (%d candidate pairs, %d scored hypotheses)" % (m, r["n_pairs"], r["n_items"]), "",
               "| bin | new triples | covered | micro recall | null | macro recall | null | mean nDCG |",
               "|---|---|---|---|---|---|---|---|"]
        for b in r["bins"]:
            if not b.get("n"):
                md.append("| %s | 0 | - | - | - | - | - | - |" % b["bin"])
            else:
                md.append("| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |" % (
                    b["bin"], b["n"], b["covered"], b["micro"], b["null_micro"], b["macro"], b["null_macro"], b["ndcg"]))
        c = r["calibration_in_the_large"]
        if c:
            md += ["", "Calibration in the large (window %s): expected %.1f vs observed %d new triples among %d scored "
                   "(observed < expected is normal: unpublished != false)." % (c["window"], c["expected"], c["observed"], c["n_items"])]
        if r["top_weights"]:
            md += ["", "Strongest standardized features: " + ", ".join("%s %+.2f" % w for w in r["top_weights"])]
        md.append("")
    md += ["Read: model >> null in every bin means the graph carries predictive signal. 'covered' bounds "
           "recall (pairs never generated as candidates cannot be predicted)."]
    write_json(os.path.join(rd, "backtest.json"), res)
    write_text(os.path.join(rd, "backtest.md"), "\n".join(md) + "\n")
    print("\n".join(md))
    log("[backtest] saved to %s" % rd)


def cmd_predict(args):
    ws = get_ws(args)
    first, support, types, _, _ = load_first(args, ws)
    cutoff = args.cutoff if args.cutoff is not None else max(first.values())
    seeds = resolve_seeds(args, ws)
    params = {k: v for k, v in vars(args).items() if k != "fn"}
    params["cutoff"] = cutoff
    label = "c%s_%s" % (cutoff, "external" if args.import_scores else args.model)
    rows = []
    if args.import_scores:
        g = TKG(first, cutoff=cutoff, support=support, types=types)
        model_meta = {"model": "external", "source": os.path.abspath(args.import_scores)}
        for row in read_table(args.import_scores):
            s, r, o = row.get("subject"), row.get("relation"), row.get("object")
            if not s or not r or not o:
                continue
            if (not args.include_connected and g.connected(s, o)) or (s, r, o) in g.first:
                continue
            sc_ = float(row.get("score") or row.get("prob") or 0)
            pr = row.get("prob")
            rows.append({"subject": s, "relation": r, "object": o, "score": sc_,
                         "prob": float(pr) if pr not in (None, "") else "", "support": "", "model": "external"})
    else:
        try:
            g, sc, model_meta = build_scorer(args, first, support, types, cutoff, seeds)
        except ValueError as e:
            die("model fitting failed: %s" % e)
        log("[predict] graph at %s: %d entities, %d triples" % (cutoff, len(g.nbrs), len(g.first)))
        pairs = candidate_pairs(g, seeds, args.seed_mode, args.include_connected, args.hub_cap, args.max_pairs)
        log("[predict] candidate pairs: %d" % len(pairs))
        for s, r, o, val, sup in score_candidates(sc, pairs, args.include_connected, args.labels_per_pair):
            if val < args.min_score:
                continue
            if args.model == "features":
                prob = val
            else:
                prob = apply_platt(sc.calibration, r, val) if sc.calibration else None
            rows.append({"subject": s, "relation": r, "object": o, "score": val,
                         "prob": prob if prob is not None else "", "support": sup, "model": args.model})
    rows.sort(key=lambda x: -x["score"])
    by_rel = defaultdict(list)
    for i, row in enumerate(rows):
        by_rel[row["relation"]].append(i)
    for idxs in by_rel.values():
        n = len(idxs)
        for rank, i in enumerate(idxs):
            rows[i]["rel_percentile"] = 1 - rank / n
    if args.min_prob is not None:
        rows = [r for r in rows if r["prob"] != "" and r["prob"] >= args.min_prob]
    if args.limit:
        rows = rows[: args.limit]
    rd = ws.new_run("predict", label, params, sys.argv[1:])
    write_json(os.path.join(rd, "model.json"), model_meta)
    write_rows(os.path.join(rd, "predictions.csv"), PRED_FIELDS,
               [{k: _fmt(v, 6) for k, v in r.items()} for r in rows])
    if args.model == "features" and not args.import_scores:
        w = describe_weights(model_meta["params"])
        log("[predict] strongest features: " + ", ".join("%s %+.2f" % x for x in w[:6]))
    print(rd)
    log("[predict] %d hypotheses -> %s/predictions.csv" % (len(rows), rd))


# --------------------------------------------------------------------------- #
# post-processing: explain, select, novelty, language branch, fuse, report
# --------------------------------------------------------------------------- #


def _enumerate_paths(g, s, o, max_len, cap, nbr_cap):
    paths = []

    def nb(x):
        ns = g.nbrs.get(x, set())
        if len(ns) > nbr_cap:
            top = sorted(ns, key=lambda n: -g.weight(x, n))[:nbr_cap]
            return top + ([o] if o in ns and o not in top else [])
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


def explain_one(g, sc, s, r, o, max_len=3, max_paths=2000, top=5, strategy="both",
                diversity=0.5, latent=False, nbr_cap=40, avoid=()):
    lab = r + ">"
    base, sup = sc.score(s, o)
    f = base.get(lab, 0.0)
    ranked = sorted(base, key=lambda k: -base[k])
    already = (s, r, o) in g.first
    cands = []
    for p in _enumerate_paths(g, s, o, max_len, max_paths, nbr_cap):
        E = {t[:3] for a, b in zip(p, p[1:]) for t in g.triples_between(a, b)}
        E.discard((s, r, o))
        suff = sc.score(s, o, only=E)[0].get(lab, 0.0)
        nec = f - sc.score(s, o, exclude=E)[0].get(lab, 0.0)
        cands.append({"path": p, "E": E, "suff": suff, "nec": nec})
    norm = f if f > 0 else 1.0
    for c in cands:
        su, ne = min(1.0, c["suff"] / norm), min(1.0, max(0.0, c["nec"]) / norm)
        c["psi"] = su if strategy == "sufficiency" else ne if strategy == "necessity" else (
            2 * su * ne / (su + ne) if su + ne > 0 else 0.0)
    cands.sort(key=lambda c: -c["psi"])
    if avoid:  # generic hubs (entities.json "exclude") make uninformative mechanisms
        specific = [c for c in cands if not set(c["path"][1:-1]) & avoid]
        cands = specific or cands
    useful = [c for c in cands if c["psi"] > 0]
    pool, chosen = (useful or cands)[: max(top * 5, top)], []
    while pool and len(chosen) < top:
        best, bval = None, -1e9
        for c in pool:
            mids, sig = set(c["path"][1:-1]), tuple(sorted(t[1] for t in c["E"]))
            pen = 0.0
            for d in chosen:
                dm = set(d["path"][1:-1])
                pen = max(pen, len(mids & dm) / max(1, len(mids | dm)) +
                          (0.3 if sig == tuple(sorted(t[1] for t in d["E"])) else 0.0))
            if c["psi"] - diversity * pen > bval:
                best, bval = c, c["psi"] - diversity * pen
        chosen.append(best)
        pool.remove(best)
    lat = []
    if latent:
        half = sorted(g.nbrs.get(s, set()) ^ g.nbrs.get(o, set()),
                      key=lambda n: -(g.weight(s, n) + g.weight(n, o)))[:300]
        for n in half:
            if n in (s, o):
                continue
            a, b = (s, n) if n in g.nbrs.get(s, set()) else (n, o)
            gap = (n, o) if a == s else (s, n)
            scs = sc.score(*gap)[0]
            if scs:
                tok, val = max(scs.items(), key=lambda x: x[1])
                lat.append({"observed": [list(t) for t in g.triples_between(a, b)],
                            "predicted": list(tok_to_triple(gap[0], gap[1], tok)), "score": val})
        lat.sort(key=lambda x: -x["score"])
        lat = lat[:top]
    return {
        "hypothesis": [s, r, o], "cutoff": g.T, "scorer": sc.kind, "score": f, "support": sup,
        "label_rank": ranked.index(lab) + 1 if lab in ranked else None, "n_labels": len(ranked),
        "already_known": already, "shortest_path": g.shortest_path_len(s, o), "paths_examined": len(cands),
        "explanations": [{"path": c["path"], "facts": [list(t) + [g.first.get(t)] for t in sorted(c["E"])],
                          "sufficiency": c["suff"], "necessity": c["nec"], "influence": c["psi"]} for c in chosen],
        "latent_paths": lat,
    }


def explanation_md(ex):
    s, r, o = ex["hypothesis"]
    out = ["### (%s, %s, %s)" % (s, r, o),
           "score=%.4f (%s) | label rank %s/%s | shortest path %s | paths examined %d" % (
               ex["score"], ex["scorer"], ex["label_rank"], ex["n_labels"], ex["shortest_path"], ex["paths_examined"])]
    if ex["already_known"]:
        out.append("> already in the graph at the cutoff - not novel")
    if not ex["explanations"]:
        out.append("No connecting paths found: no structural support (speculative).")
    for i, e in enumerate(ex["explanations"], 1):
        out.append("- **E%d** influence %.3f (sufficiency %.4f, necessity %.4f): %s" % (
            i, e["influence"], e["sufficiency"], e["necessity"], " -- ".join(e["path"])))
        for f in e["facts"]:
            out.append("  - (%s, %s, %s)%s" % (f[0], f[1], f[2], " [%s]" % f[3] if f[3] is not None else ""))
    for l in ex.get("latent_paths") or []:
        obs = "; ".join("(%s, %s, %s) [%s]" % tuple(x) for x in l["observed"])
        out.append("- latent: %s + predicted (%s, %s, %s) p=%.3f" % ((obs,) + tuple(l["predicted"]) + (l["score"],)))
    return "\n".join(out)


def _run_rows(rd, prefer=("fused.csv", "shortlist.csv", "predictions.csv"), source=None):
    names = [source] if source else list(prefer)
    for n in names:
        p = os.path.join(rd, n)
        if os.path.exists(p):
            return read_table(p), n
    die("none of %s in %s" % (", ".join(names), rd))


def cmd_explain(args):
    ws = get_ws(args)
    first, support, types, _, ent_cfg = load_first(args, ws)
    avoid = set(ent_cfg.get("exclude") or []) if not args.allow_hubs else set()
    rd = ws.run_dir(args.run, kind="predict")
    g, sc, cutoff = scorer_from_run(ws, rd, args, first, support, types)
    if args.subject:
        targets = [(args.subject, args.relation, args.object)]
    else:
        rows, name = _run_rows(rd, ("shortlist.csv",))
        targets = [(r["subject"], r["relation"], r["object"]) for r in rows[: args.limit]]
    os.makedirs(os.path.join(rd, "explanations"), exist_ok=True)
    allmd, allj = [], []
    for s, r, o in targets:
        if s not in g.nbrs or o not in g.nbrs:
            log("[explain] skip (%s, %s, %s): entity absent at cutoff %s" % (s, r, o, cutoff))
            continue
        ex = explain_one(g, sc, s, r, o, args.max_len, args.max_paths, args.top, args.strategy,
                         args.diversity, args.latent, args.nbr_cap, avoid)
        allj.append(ex)
        allmd.append(explanation_md(ex))
    write_json(os.path.join(rd, "explanations", "explanations.json"), allj)
    text = "# Explanations (PHELInE-style)\n\n" + "\n\n".join(allmd) + "\n"
    write_text(os.path.join(rd, "explanations", "explanations.md"), text)
    print(text)


def cmd_select(args):
    ws = get_ws(args)
    first, support, types, _, ent_cfg = load_first(args, ws)
    rd = ws.run_dir(args.run, kind="predict")
    man = read_json(os.path.join(rd, "manifest.json"))
    cutoff = man["params"]["cutoff"]
    g = TKG(first, cutoff=cutoff, support=support, types=types)
    rows, src = _run_rows(rd, ("fused.csv", "predictions.csv"), args.source)
    col = args.score_col or ("fused" if src == "fused.csv" else ("prob" if rows and rows[0].get("prob") else "score"))
    fields = list(rows[0].keys()) if rows else []
    fields += [f for f in ("novelty", "recency_subject", "recency_object", "shortest_path") if f not in fields]
    log_ = [("input (%s)" % src, len(rows), "ranked by %s" % col)]
    nov = {}
    for c in ws.novelty_checks():
        nov[pkey(c.get("subject", ""), c.get("object", ""))] = c.get("status")
    seen, stage = set(), []
    for p in rows:
        k = (p["subject"], p["relation"], p["object"])
        if k in seen:
            continue
        seen.add(k)
        if (args.allow_connected and k in g.first) or (not args.allow_connected and g.connected(p["subject"], p["object"])):
            continue
        try:
            p["_score"] = float(p[col])
        except (TypeError, ValueError, KeyError):
            continue
        p["novelty"] = nov.get(pkey(p["subject"], p["object"]), "unchecked")
        stage.append(p)
    log_.append(("prepare", len(stage), "dedupe, drop relations already in the graph"))
    before = len(stage)
    stage = [p for p in stage if p["novelty"] != "known"]
    if before != len(stage):
        log_.append(("novelty", len(stage), "dropped hypotheses already reported"))
    if args.min_score is not None:
        stage = [p for p in stage if p["_score"] >= args.min_score]
        log_.append(("threshold", len(stage), "%s >= %s" % (col, args.min_score)))
    excl = set(ent_cfg.get("exclude", []) or []) | set(read_list(args.exclude_entities) or [])
    if excl:
        stage = [p for p in stage if p["subject"] not in excl and p["object"] not in excl]
        log_.append(("exclude", len(stage), "%d excluded entities" % len(excl)))
    stage.sort(key=lambda p: -p["_score"])
    keep, per = set(), defaultdict(int)
    for i, p in enumerate(stage):
        for e in (p["subject"], p["object"]):
            if per[e] < args.top_k:
                per[e] += 1
                keep.add(i)
    stage = [p for i, p in enumerate(stage) if i in keep]
    log_.append(("top-k per entity", len(stage), "k=%d (union over subject/object)" % args.top_k))
    for p in stage:
        p["recency_subject"] = g.recency(p["subject"])
        p["recency_object"] = g.recency(p["object"])
        p["shortest_path"] = g.shortest_path_len(p["subject"], p["object"], args.max_path)
    if args.min_path:
        stage = [p for p in stage if p["shortest_path"] is None or p["shortest_path"] >= args.min_path]
        log_.append(("min path", len(stage), "shortest path >= %d" % args.min_path))
    if args.limit:
        stage = stage[: args.limit]
        log_.append(("limit", len(stage), "top %d" % args.limit))
    write_rows(os.path.join(rd, "shortlist.csv"), fields, stage)
    rep = ["# Hypothesis selection", "", "| step | output | note |", "|---|---|---|"]
    rep += ["| %s | %d | %s |" % x for x in log_]
    pl = Counter(p["shortest_path"] for p in stage)
    rep += ["", "| shortest path | count |", "|---|---|"]
    rep += ["| %s | %d |" % (k if k is not None else "none", v)
            for k, v in sorted(pl.items(), key=lambda x: (x[0] is None, x[0] or 0))]
    write_text(os.path.join(rd, "selection.md"), "\n".join(rep) + "\n")
    print("\n".join(rep))
    log("[select] %d hypotheses -> %s/shortlist.csv" % (len(stage), rd))


def _quick_paths(g, s, o, k=3):
    out = []
    for n in sorted(g.nbrs.get(s, set()) & g.nbrs.get(o, set()),
                    key=lambda n: -(g.weight(s, n) * g.weight(n, o)))[:k]:
        facts = [t for t in g.triples_between(s, n)] + [t for t in g.triples_between(n, o)]
        out.append(["(%s, %s, %s) [%s]" % t for t in facts])
    return out


JUDGE_PROMPT = """# Language-branch judging protocol ({version})

You are the language branch of a Hakken-style predictor. For EACH item in your batch file:

1. Evidence (separated, pre-cutoff): gather 5-20 statements about the SUBJECT alone and about
   the OBJECT alone, published in or before {{cutoff}}. Preferred tool:
     python3 <skill>/scripts/hakken.py evidence --entity "<SUBJECT>" --before <cutoff> --exclude-term "<OBJECT>" --context "<entity type terms>"
   (and vice versa). Never search for both entities together and never use knowledge of
   work published after the cutoff. If you recognise the pair as already published, say so
   in "risk" and set "leak_suspected": true instead of using that knowledge.
2. Read the graph evidence paths and alternative relation labels provided in the item.
3. Judge: is there a coherent mechanism? Is the relation TYPE and DIRECTION the most plausible
   among the listed relation definitions?
4. Output ONE JSON object per item (JSON Lines), nothing else:
   {{"id": "<id>", "subject": "...", "relation": "...", "object": "...", "cutoff": <cutoff>,
    "score": <0.0-1.0 plausibility>, "best_relation": "<label or same>",
    "mechanism": "<one sentence>", "risk": "<main reason it could be wrong>",
    "evidence": ["PMID:..", "openalex:.."], "leak_suspected": false,
    "model": "<your model name>", "prompt_version": "{version}"}}
Score anchors: 0.1 no plausible mechanism; 0.3 weak/indirect; 0.5 plausible but generic;
0.7 specific mechanism supported by both entities' evidence; 0.9 strong, specific, consistent paths.

Save results to a .jsonl file and register them with:
  python3 <skill>/scripts/hakken.py record judgments <file.jsonl>
"""


def cmd_judge_queue(args):
    ws = get_ws(args)
    first, support, types, rel_cfg, _ = load_first(args, ws)
    rd = ws.run_dir(args.run, kind="predict")
    cutoff = read_json(os.path.join(rd, "manifest.json"))["params"]["cutoff"]
    g = TKG(first, cutoff=cutoff, support=support, types=types)
    rows, src = _run_rows(rd, ("shortlist.csv", "predictions.csv"), args.source)
    preds, _ = _run_rows(rd, ("predictions.csv",))
    alts = defaultdict(list)
    for p in preds:
        alts[pkey(p["subject"], p["object"])].append("(%s, %s, %s) %s" % (p["subject"], p["relation"], p["object"], p.get("prob") or p["score"]))
    done = {(j.get("subject"), j.get("relation"), j.get("object"), str(j.get("cutoff")))
            for j in ws.judgments() if j.get("prompt_version") == PROMPT_VERSION}
    defs = rel_cfg.get("definitions") or {}
    items = []
    for p in rows:
        k = (p["subject"], p["relation"], p["object"], str(cutoff))
        if k in done:
            continue
        items.append({"id": hyp_key(*k[:3]), "subject": p["subject"], "relation": p["relation"],
                      "object": p["object"], "cutoff": cutoff,
                      "graph_score": p.get("prob") or p.get("score"),
                      "relation_definition": defs.get(p["relation"], ""),
                      "alternative_labels": alts[pkey(p["subject"], p["object"])][:5],
                      "graph_evidence_paths": _quick_paths(g, p["subject"], p["object"]),
                      "subject_type": types.get(p["subject"], ""), "object_type": types.get(p["object"], "")})
        if len(items) >= args.top:
            break
    qd = os.path.join(rd, "judge")
    os.makedirs(qd, exist_ok=True)
    nb = 0
    for i in range(0, len(items), args.batch_size):
        nb += 1
        with open(os.path.join(qd, "batch_%03d.jsonl" % nb), "w", encoding="utf-8") as fh:
            for it in items[i:i + args.batch_size]:
                fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    write_text(os.path.join(qd, "PROMPT.md"), JUDGE_PROMPT.format(version=PROMPT_VERSION).replace("{cutoff}", str(cutoff)))
    if defs:
        write_json(os.path.join(qd, "relation_definitions.json"), defs)
    print(qd)
    log("[judge-queue] %d items (from %s) in %d batch file(s); %d already judged were skipped"
        % (len(items), src, nb, len(done)))


def cmd_record(args):
    ws = get_ws(args)
    path = args.file
    recs = read_jsonl(path) if path != "-" else [json.loads(l) for l in sys.stdin if l.strip()]
    ok, bad = [], 0
    ts = now_iso()
    for r in recs:
        if args.kind == "judgments":
            try:
                r["score"] = float(r["score"])
                assert 0.0 <= r["score"] <= 1.0
                assert r.get("subject") and r.get("relation") and r.get("object") and r.get("cutoff") is not None
            except (KeyError, ValueError, TypeError, AssertionError):
                bad += 1
                continue
            r.setdefault("prompt_version", PROMPT_VERSION)
        else:
            if not (r.get("subject") and r.get("object") and r.get("status") in
                    ("novel", "known", "co-mentioned", "unclear", "pending")):
                bad += 1
                continue
        r["recorded_at"] = ts
        ok.append(r)
    (ws.append_judgments if args.kind == "judgments" else ws.append_novelty)(ok)
    log("[record] %d %s appended, %d invalid skipped" % (len(ok), args.kind, bad))


def cmd_novelty(args):
    ws = get_ws(args)
    first, support, types, _, _ = load_first(args, ws)
    rd = ws.run_dir(args.run, kind="predict")
    cutoff = read_json(os.path.join(rd, "manifest.json"))["params"]["cutoff"]
    rows, src = _run_rows(rd, ("shortlist.csv", "predictions.csv"), args.source)
    rows = rows[: args.top]
    pair_years = defaultdict(list)
    for (s, r, o), y in first.items():
        pair_years[pkey(s, o)].append((y, r))
    recs, queue = [], []
    for p in rows:
        s, r, o = p["subject"], p["relation"], p["object"]
        ys = pair_years.get(pkey(s, o))
        if ys:
            y, rel = min(ys)
            recs.append({"subject": s, "relation": r, "object": o, "status": "known", "method": "ledger",
                         "note": "pair recorded in ledger in %s as %s (after cutoff %s)" % (y, rel, cutoff),
                         "cutoff": cutoff})
        else:
            queue.append({"subject": s, "relation": r, "object": o,
                          "suggested_queries": ['"%s" AND "%s"' % (s, o)]})
    ext = []
    if args.external and queue:
        from .sources import openalex_search, pubmed_search
        check_external(ws, "external_novelty_search", args.confirmed, "external novelty search")
        for q in queue:
            s, o = q["subject"], q["object"]
            if args.external == "pubmed":
                n, ids = pubmed_search('"%s"[tiab] AND "%s"[tiab]' % (s, o), retmax=5)
                hits = ["PMID:%s" % i for i in ids]
            else:
                n, docs = openalex_search('"%s" "%s"' % (s, o), per_page=5)
                hits = [d["id"] for d in docs]
            ext.append({"subject": s, "relation": q["relation"], "object": o,
                        "status": "novel" if n == 0 else "co-mentioned", "method": "external:" + args.external,
                        "query": q["suggested_queries"][0], "hits": n, "examples": hits[:5], "cutoff": cutoff,
                        "note": "co-mention is not proof of the relation; review examples" if n else ""})
        queue = []
    if args.mark_pending and queue:
        ext += [{"subject": q["subject"], "relation": q["relation"], "object": q["object"], "status": "pending",
                 "method": "deferred", "note": "external check needs user approval", "cutoff": cutoff} for q in queue]
    ws.append_novelty(recs + ext)
    with open(os.path.join(rd, "novelty_queue.jsonl"), "w", encoding="utf-8") as fh:
        for q in queue:
            fh.write(json.dumps(q, ensure_ascii=False) + "\n")
    c = Counter(x["status"] for x in recs + ext)
    print(json.dumps({"checked": len(rows), "statuses": dict(c), "awaiting_external_check": len(queue),
                      "policy_external_novelty_search": ws.policy("external_novelty_search", "ask")}, indent=1))


def cmd_fuse(args):
    ws = get_ws(args)
    rd = ws.run_dir(args.run, kind="predict")
    cutoff = read_json(os.path.join(rd, "manifest.json"))["params"]["cutoff"]
    rows, src = _run_rows(rd, ("predictions.csv",))
    rel_alpha = (ws.relations_cfg().get("fusion_alpha") or {})
    latest = {}
    for j in ws.judgments():
        if not args.any_cutoff and str(j.get("cutoff")) != str(cutoff):
            continue
        if j.get("leak_suspected") and not args.keep_leaky:
            continue
        latest[(j["subject"], j["relation"], j["object"])] = j
    out, nj = [], 0
    for p in rows:
        k = (p["subject"], p["relation"], p["object"])
        if p.get("prob") not in (None, ""):
            pg = float(p["prob"])
        else:
            pg = min(0.99, max(0.01, float(p.get("rel_percentile") or 0.5)))
        j = latest.get(k)
        a = float(rel_alpha.get(p["relation"], args.alpha))
        if j:
            nj += 1
            pl = float(j["score"])
            fused = sigmoid(a * logit(pg) + (1 - a) * logit(pl))
        else:
            pl, fused = "", pg
        q = dict(p)
        q.update({"graph_prob": round(pg, 6), "llm_score": pl, "alpha": a if j else "",
                  "fused": round(fused, 6), "judged": 1 if j else 0,
                  "best_relation": (j or {}).get("best_relation", ""), "mechanism": (j or {}).get("mechanism", "")})
        out.append(q)
    out.sort(key=lambda x: (-x["judged"] if args.judged_first else 0, -x["fused"]))
    fields = list(rows[0].keys()) + ["graph_prob", "llm_score", "alpha", "fused", "judged", "best_relation", "mechanism"] if rows else []
    write_rows(os.path.join(rd, "fused.csv"), fields, out)
    log("[fuse] %d hypotheses, %d with language-branch judgments (alpha=%s) -> %s/fused.csv" % (len(out), nj, args.alpha, rd))


def cmd_report(args):
    ws = get_ws(args)
    rd = ws.run_dir(args.run, kind="predict")
    man = read_json(os.path.join(rd, "manifest.json"))
    model = read_json(os.path.join(rd, "model.json"), {}) or {}
    p = man["params"]
    bt = None
    for r in reversed(ws.runs()):
        m2 = read_json(ws.p("runs", r, "manifest.json"), {})
        if m2.get("kind") == "backtest":
            bt = ws.p("runs", r)
            break
    L = ["# Hakken discovery report: %s" % (ws.manifest.get("goal") or "<topic>"), "",
         "## 1. Setup", "",
         "- Workspace: `%s`; run `%s`" % (ws.root, man["run_id"]),
         "- Knowledge cutoff: %s; graph branch: %s%s" % (
             p["cutoff"], model.get("model"),
             " (logistic model on temporal network features fitted on (%s, %s])" % (p["cutoff"] - p.get("window", 0), p["cutoff"])
             if model.get("model") == "features" else ""),
         "- Ledger: %s observations (sha256 %s...)" % (man["inputs"]["ledger_rows"], (man["inputs"]["ledger_sha256"] or "")[:12]),
         "- Reproduce: `%s`" % man["command"], "",
         "## 2. Back-test", ""]
    if bt and os.path.exists(os.path.join(bt, "backtest.md")):
        with open(os.path.join(bt, "backtest.md"), encoding="utf-8") as fh:
            L += ["(from run `%s`)" % os.path.basename(bt), "", fh.read().replace("# Back-test", "### Back-test")]
    else:
        L.append("_No back-test run found - run `backtest` before trusting predictions._")
    L += ["", "## 3. Selection", ""]
    if os.path.exists(os.path.join(rd, "selection.md")):
        with open(os.path.join(rd, "selection.md"), encoding="utf-8") as fh:
            L.append(fh.read().replace("# Hypothesis selection", "").strip())
    L += ["", "## 4. Hypotheses", ""]
    rows = read_table(os.path.join(rd, "shortlist.csv")) if os.path.exists(os.path.join(rd, "shortlist.csv")) else []
    exps = {tuple(e["hypothesis"]): e for e in (read_json(os.path.join(rd, "explanations", "explanations.json"), []) or [])}
    judg = {(j["subject"], j["relation"], j["object"]): j for j in ws.judgments() if str(j.get("cutoff")) == str(p["cutoff"])}
    nov = {}
    for c in ws.novelty_checks():
        nov[pkey(c["subject"], c["object"])] = c
    for i, r in enumerate(rows[: args.top], 1):
        k = (r["subject"], r["relation"], r["object"])
        j, e, n = judg.get(k), exps.get(k), nov.get(pkey(r["subject"], r["object"]))
        L += ["### H%d. (%s, %s, %s)" % ((i,) + k), "",
              "- Graph branch: prob %s | score %s | relation percentile %s" % (r.get("graph_prob") or r.get("prob") or "-", r.get("score"), r.get("rel_percentile")),
              "- Language branch: %s" % ("score %.2f - %s (risk: %s; judged by %s)" % (j["score"], j.get("mechanism", ""), j.get("risk", ""), j.get("model", "?")) if j else "not judged"),
              "- Fused: %s" % (r.get("fused") or "-"),
              "- Novelty: %s" % ("%s via %s %s" % (n["status"], n.get("method"), n.get("note", "")) if n else "unchecked"),
              "- Context: recency %s / %s; shortest path %s" % (r.get("recency_subject"), r.get("recency_object"), r.get("shortest_path"))]
        if e:
            for x, ex in enumerate(e["explanations"][:3], 1):
                L.append("- Evidence path E%d (suff %.3f, nec %.3f): %s" % (
                    x, ex["sufficiency"], ex["necessity"],
                    "; ".join("(%s, %s, %s) [%s]" % tuple(f) for f in ex["facts"])))
        L += ["- Proposed mechanism: <!-- agent -->", "- Why it could be wrong: <!-- agent -->",
              "- Minimal validation experiment: <!-- agent: system, perturbation of subject, readout of object, controls, expected direction -->", ""]
    L += ["## 5. Caveats", "",
          "- Scores measure literature support, not truth; unobserved relations are unlabeled, not false.",
          "- The graph branch is a feature-based surrogate of THiGER, not the THiGERLLM neural model.",
          "- Extraction misses negation, conditions and quantities; polarity predictions are direction-uncertain unless paths agree.",
          "- Language-branch judgments may be contaminated by the LLM's knowledge of post-cutoff literature.", ""]
    write_text(os.path.join(rd, "report.md"), "\n".join(L) + "\n")
    print(os.path.join(rd, "report.md"))


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hakken.py", description="Hakken-style knowledge prediction (stdlib only).")
    ap.add_argument("--version", action="version", version="hakken-discovery " + __version__)
    ap.add_argument("--workspace", "-w", help="workspace dir (default: $HAKKEN_WORKSPACE or nearest hakken/)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def P(name, fn, help_):
        p = sub.add_parser(name, help=help_)
        p.set_defaults(fn=fn)
        p.add_argument("--workspace", "-w", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
        return p

    p = P("init", cmd_init, "create a workspace (default ./hakken)")
    p.add_argument("path", nargs="?")
    p.add_argument("--goal")
    p.add_argument("--domain")
    p.add_argument("--time-unit", default="year")
    p.add_argument("--git-policy", choices=["private", "standard", "none"], default="private",
                   help="private: track config+ledger only (default); standard: also judgments/novelty; none: ignore all")
    p.add_argument("--novelty-search", choices=["ask", "allow", "deny"], default="ask")
    p.add_argument("--evidence-search", choices=["allow", "ask", "deny"], default="allow")
    p.add_argument("--force", action="store_true")

    P("status", cmd_status, "show workspace contents")

    p = P("ingest", cmd_ingest, "append observations (CSV/TSV with subject,relation,object,year[,source,evidence,...])")
    p.add_argument("file")
    p.add_argument("--extractor", required=True, help="provenance, e.g. user, agent:<model>, semmeddb")
    p.add_argument("--source-col")

    p = P("fetch-pubtator", cmd_fetch_pubtator, "biomedical: fetch dated relations from PubTator3 into the ledger")
    p.add_argument("--query", action="append", required=True, help="e.g. @GENE_TP53, @DISEASE_Aging or free text; repeatable")
    p.add_argument("--max-docs", type=int, default=500, help="documents per query (without --year-from)")
    p.add_argument("--year-from", type=int, help="sample documents per publication year from this year")
    p.add_argument("--year-to", type=int, help="last year for per-year sampling")
    p.add_argument("--per-year", type=int, default=30, help="documents per query and year")
    p.add_argument("--confirmed", action="store_true")

    p = P("evidence", cmd_evidence, "single-entity sentences published before a year (language-branch input)")
    p.add_argument("--entity", required=True)
    p.add_argument("--before", type=int)
    p.add_argument("--source", choices=["pubmed", "openalex"], default="pubmed")
    p.add_argument("--exclude-term", action="append", help="drop sentences mentioning this (the other entity)")
    p.add_argument("--context", help='disambiguating terms ANDed to the query, e.g. "gene OR protein" (never the other entity)')
    p.add_argument("--docs", type=int, default=25)
    p.add_argument("--max", type=int, default=20)
    p.add_argument("--confirmed", action="store_true")

    for name, fn, h in (("stats", cmd_stats, "graph statistics"), ("export-graph", cmd_export_graph, "write the cleaned graph")):
        p = P(name, fn, h)
        p.add_argument("--triples", help="use this file instead of the workspace ledger")
        p.add_argument("--config", help="cleaning config JSON overriding workspace config")
        p.add_argument("--cutoff", type=int)
        if name == "stats":
            p.add_argument("--top", type=int, default=15)
        else:
            p.add_argument("-o", "--output", default="-")

    p = P("backtest", cmd_backtest, "historic evaluation vs a random-empirical-score null")
    p.add_argument("--triples")
    p.add_argument("--config")
    p.add_argument("--cutoff", type=int, required=True)
    p.add_argument("--bin", type=int, default=2)
    p.add_argument("--bins", type=int, default=5)
    p.add_argument("--top-frac", type=float, default=0.1)
    p.add_argument("--null-runs", type=int, default=5)
    p.add_argument("--min-support", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    model_args(p)
    p.set_defaults(model="both")
    p._option_string_actions["--model"].choices = ["features", "heuristic", "both"]

    p = P("predict", cmd_predict, "score novel hypotheses into a new run")
    p.add_argument("--triples")
    p.add_argument("--config")
    p.add_argument("--cutoff", type=int, help="knowledge cutoff (default: latest time in data)")
    p.add_argument("--labels-per-pair", type=int, default=3)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--min-prob", type=float)
    p.add_argument("--limit", type=int, default=50000, help="keep the top N hypotheses (0 = all)")
    p.add_argument("--import-scores", help="CSV subject,relation,object,score[,prob] from an external model")
    model_args(p)

    p = P("explain", cmd_explain, "PHELInE-style path explanations (one hypothesis or the shortlist)")
    p.add_argument("--run", default="latest")
    p.add_argument("--triples")
    p.add_argument("--config")
    p.add_argument("--subject", "-s")
    p.add_argument("--relation", "-r")
    p.add_argument("--object", "-O")
    p.add_argument("--limit", type=int, default=20, help="shortlist rows to explain when no hypothesis given")
    p.add_argument("--max-len", type=int, default=3)
    p.add_argument("--max-paths", type=int, default=2000)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--strategy", choices=["both", "sufficiency", "necessity"], default="both")
    p.add_argument("--diversity", type=float, default=0.5)
    p.add_argument("--latent", action="store_true")
    p.add_argument("--nbr-cap", type=int, default=40)
    p.add_argument("--allow-hubs", action="store_true", help="also show paths through entities.json exclude hubs")

    p = P("select", cmd_select, "hypothesis-selection pipeline -> shortlist.csv")
    p.add_argument("--run", default="latest")
    p.add_argument("--triples")
    p.add_argument("--config")
    p.add_argument("--source", choices=["fused.csv", "predictions.csv"])
    p.add_argument("--score-col")
    p.add_argument("--min-score", type=float)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--min-path", type=int, default=0)
    p.add_argument("--max-path", type=int, default=6)
    p.add_argument("--limit", type=int)
    p.add_argument("--exclude-entities")
    p.add_argument("--allow-connected", action="store_true")

    p = P("judge-queue", cmd_judge_queue, "export hypotheses for language-branch judging (sub-agents)")
    p.add_argument("--run", default="latest")
    p.add_argument("--triples")
    p.add_argument("--config")
    p.add_argument("--source", choices=["shortlist.csv", "predictions.csv"])
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=10)

    p = P("record", cmd_record, "append judgments or novelty checks (JSONL) to the workspace")
    p.add_argument("kind", choices=["judgments", "novelty"])
    p.add_argument("file", help="JSONL file or - for stdin")

    p = P("novelty", cmd_novelty, "novelty checks: ledger (local) and optional external co-mention search")
    p.add_argument("--run", default="latest")
    p.add_argument("--triples")
    p.add_argument("--config")
    p.add_argument("--source", choices=["shortlist.csv", "predictions.csv"])
    p.add_argument("--top", type=int, default=50)
    p.add_argument("--external", choices=["pubmed", "openalex"])
    p.add_argument("--confirmed", action="store_true")
    p.add_argument("--mark-pending", action="store_true", help="record unchecked pairs as pending (autonomous modes)")

    p = P("fuse", cmd_fuse, "label-wise logit ensemble of graph probability and language-branch score")
    p.add_argument("--run", default="latest")
    p.add_argument("--alpha", type=float, default=0.5, help="weight of the graph branch (per-relation override: fusion_alpha in relations.json)")
    p.add_argument("--any-cutoff", action="store_true")
    p.add_argument("--keep-leaky", action="store_true", help="keep judgments flagged leak_suspected")
    p.add_argument("--judged-first", action="store_true")

    p = P("report", cmd_report, "assemble report.md for a run")
    p.add_argument("--run", default="latest")
    p.add_argument("--top", type=int, default=10)

    args = ap.parse_args(argv)
    args.fn(args)
