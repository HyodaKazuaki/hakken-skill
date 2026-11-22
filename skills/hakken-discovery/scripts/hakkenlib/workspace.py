"""Project workspace: where all knowledge data lives (never inside the skill directory).

Layout (created by `hakken.py init`):

    hakken/
      hakken.json            workspace manifest: schema, goal, time unit, policies
      README.md              what each file is (for humans and other agents)
      .gitignore             scoped ignore rules (the project's own .gitignore is never edited)
      config/relations.json  relation vocabulary, definitions, rename/swap/drop, polarity groups
      config/entities.json   synonyms, entities to exclude from selection
      config/seeds.txt       optional focus entities
      ledger/observations.csv  SOURCE OF TRUTH: append-only literature observations + provenance
      judgments/llm.jsonl    append-only language-branch judgments (agent / sub-agents)
      novelty/checks.jsonl   append-only novelty checks (ledger / external search / pending)
      runs/<run-id>/         derived, reproducible outputs with manifest.json

Discovery order: --workspace, $HAKKEN_WORKSPACE, then walk up from CWD looking for
hakken/hakken.json (or a directory that itself contains hakken.json).
Download caches go to $HAKKEN_CACHE or $XDG_CACHE_HOME/hakken (default ~/.cache/hakken).
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import os
import re

from . import WORKSPACE_SCHEMA, __version__
from .common import (append_jsonl, die, file_lock, log, now_iso, parse_year,
                     read_json, read_jsonl, write_json, write_text)

LEDGER_FIELDS = ["subject", "relation", "object", "year", "source", "evidence",
                 "subject_type", "object_type", "extractor", "ingested_at"]

GITIGNORE = {
    # privacy-first default: track only curated config and the literature ledger
    "private": "# managed by hakken - generated hypotheses stay local\nruns/\njudgments/\nnovelty/\n*.lock\n.tmp-*\n",
    # track everything that is not reproducible; ignore only derived runs
    "standard": "# managed by hakken - derived outputs are reproducible\nruns/\n*.lock\n.tmp-*\n",
    # keep the whole workspace out of version control
    "none": "# managed by hakken - workspace not version controlled\n*\n",
}

DEFAULT_RELATIONS = {
    "_doc": "Relation vocabulary. Names are UPPER_SNAKE, directed, active voice. "
            "definitions are shown to language-branch judges. rename/swap/drop are applied "
            "when the graph is built; polarity groups follow Hakken Table E2.",
    "definitions": {},
    "rename": {},
    "swap": {},
    "drop": [],
    "keep_only": [],
    "polarity": [],
    "upper_snake": True,
    "min_relation_count": 1,
}

DEFAULT_ENTITIES = {
    "_doc": "synonyms map variant -> canonical name. exclude lists overly generic hubs "
            "that should never appear in shortlisted hypotheses.",
    "synonyms": {},
    "exclude": [],
    "lowercase": False,
}

README = """# Hakken workspace

Created by the `hakken-discovery` agent skill (v{version}). All knowledge data for this
project lives here; the skill directory only contains code.

| Path | What | Version control |
|---|---|---|
| `hakken.json` | goal, time unit, policies, schema version | tracked |
| `config/` | relation vocabulary, synonyms, exclusions, seeds (human-curated) | tracked |
| `ledger/observations.csv` | **source of truth**: literature observations with year, source and extractor (append-only) | tracked |
| `judgments/llm.jsonl` | language-branch plausibility judgments (append-only) | per git policy |
| `novelty/checks.jsonl` | novelty checks incl. pending external searches (append-only) | per git policy |
| `runs/<id>/` | derived outputs (graph snapshot, model, predictions, explanations, report) | ignored |

Git policy: **{policy}** (see `.gitignore`). Generated hypotheses are unpublished research
ideas; keep `judgments/`, `novelty/` and `runs/` out of public repositories.

Regenerate any run from the ledger + config with the command stored in its `manifest.json`.
"""


def cache_dir(sub=""):
    base = os.environ.get("HAKKEN_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "hakken")
    p = os.path.join(base, sub) if sub else base
    os.makedirs(p, exist_ok=True)
    return p


def _is_ws(path):
    return os.path.isfile(os.path.join(path, "hakken.json"))


def find_workspace(explicit=None, start=None):
    if explicit:
        p = os.path.abspath(explicit)
        if _is_ws(p):
            return p
        if _is_ws(os.path.join(p, "hakken")):
            return os.path.join(p, "hakken")
        return None
    env = os.environ.get("HAKKEN_WORKSPACE")
    if env:
        return find_workspace(env)
    cur = os.path.abspath(start or os.getcwd())
    while True:
        for cand in (os.path.join(cur, "hakken"), cur):
            if _is_ws(cand):
                return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            return None
        cur = parent


class Workspace:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.manifest = read_json(self.p("hakken.json"), {})
        sv = self.manifest.get("schema_version", 0)
        if sv > WORKSPACE_SCHEMA:
            die("workspace schema %s is newer than this skill (%s); update the skill" % (sv, WORKSPACE_SCHEMA))

    # ---- paths ------------------------------------------------------------ #
    def p(self, *parts):
        return os.path.join(self.root, *parts)

    @property
    def ledger_path(self):
        return self.p("ledger", "observations.csv")

    @property
    def judgments_path(self):
        return self.p("judgments", "llm.jsonl")

    @property
    def novelty_path(self):
        return self.p("novelty", "checks.jsonl")

    def policy(self, key, default=None):
        return (self.manifest.get("policy") or {}).get(key, default)

    # ---- config ------------------------------------------------------------ #
    def relations_cfg(self):
        cfg = dict(DEFAULT_RELATIONS)
        cfg.update(read_json(self.p("config", "relations.json"), {}) or {})
        return cfg

    def entities_cfg(self):
        cfg = dict(DEFAULT_ENTITIES)
        cfg.update(read_json(self.p("config", "entities.json"), {}) or {})
        return cfg

    def seeds(self):
        path = self.p("config", "seeds.txt")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            s = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        return s or None

    # ---- ledger ------------------------------------------------------------ #
    def read_ledger(self):
        if not os.path.exists(self.ledger_path):
            return []
        with open(self.ledger_path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    def ledger_triples(self):
        """Ledger rows -> (s, r, o, year, extra) tuples (raw, before config is applied)."""
        out = []
        for row in self.read_ledger():
            try:
                y = parse_year(row["year"])
            except (ValueError, KeyError):
                continue
            out.append((row["subject"], row["relation"], row["object"], y, row))
        return out

    def append_ledger(self, rows, extractor):
        """Append observations; exact duplicates (s, r, o, year, source) are skipped."""
        os.makedirs(self.p("ledger"), exist_ok=True)
        with file_lock(self.ledger_path + ".lock"):
            seen = set()
            exists = os.path.exists(self.ledger_path)
            if exists:
                for r in self.read_ledger():
                    seen.add((r["subject"], r["relation"], r["object"], str(r["year"]), r.get("source", "")))
            added = 0
            ts = now_iso()
            with open(self.ledger_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=LEDGER_FIELDS, extrasaction="ignore")
                if not exists:
                    w.writeheader()
                for r in rows:
                    k = (r["subject"], r["relation"], r["object"], str(r["year"]), r.get("source", ""))
                    if k in seen:
                        continue
                    seen.add(k)
                    rec = {f: r.get(f, "") for f in LEDGER_FIELDS}
                    rec["extractor"] = r.get("extractor") or extractor
                    rec["ingested_at"] = ts
                    w.writerow(rec)
                    added += 1
        return added

    # ---- append-only logs -------------------------------------------------- #
    def judgments(self):
        return read_jsonl(self.judgments_path)

    def novelty_checks(self):
        return read_jsonl(self.novelty_path)

    def append_judgments(self, recs):
        append_jsonl(self.judgments_path, recs)

    def append_novelty(self, recs):
        append_jsonl(self.novelty_path, recs)

    # ---- runs --------------------------------------------------------------- #
    def runs(self):
        d = self.p("runs")
        if not os.path.isdir(d):
            return []
        return sorted(x for x in os.listdir(d) if os.path.isfile(os.path.join(d, x, "manifest.json")))

    def run_dir(self, run_id=None, kind=None):
        """Resolve a run id ('latest' or None -> newest run, optionally of a given kind)."""
        runs = self.runs()
        if run_id in (None, "latest"):
            cands = runs
            if kind:
                cands = [r for r in runs if read_json(self.p("runs", r, "manifest.json"), {}).get("kind") == kind]
            if not cands:
                die("no %sruns in %s - run `predict` first" % (kind + " " if kind else "", self.root))
            return self.p("runs", cands[-1])
        d = self.p("runs", run_id)
        if not os.path.isfile(os.path.join(d, "manifest.json")):
            die("run %r not found (have: %s)" % (run_id, ", ".join(runs[-5:]) or "none"))
        return d

    def new_run(self, kind, label, params, argv):
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        rid = "%s_%s_%s" % (ts, kind, re.sub(r"[^A-Za-z0-9_.-]+", "-", label))[:80]
        d = self.p("runs", rid)
        n = 1
        while os.path.exists(d):
            n += 1
            d = self.p("runs", "%s-%d" % (rid, n))
        os.makedirs(d)
        man = {
            "run_id": os.path.basename(d), "kind": kind, "created_at": now_iso(),
            "skill_version": __version__, "params": params,
            "inputs": {"ledger_sha256": self.ledger_hash(), "ledger_rows": len(self.read_ledger()),
                       "relations_cfg_sha256": _sha(self.p("config", "relations.json")),
                       "entities_cfg_sha256": _sha(self.p("config", "entities.json"))},
            "command": "python3 <skill>/scripts/hakken.py " + " ".join(argv),
        }
        write_json(os.path.join(d, "manifest.json"), man)
        return d

    def ledger_hash(self):
        return _sha(self.ledger_path)


def _sha(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def init_workspace(path, goal="", domain="", time_unit="year", git_policy="private",
                   external_novelty_search="ask", entity_evidence_search="allow", force=False):
    root = os.path.abspath(path)
    if _is_ws(root) and not force:
        log("[init] workspace already exists at %s (use --force to rewrite manifest/README/.gitignore)" % root)
        return Workspace(root)
    for sub in ("config", "ledger", "judgments", "novelty", "runs"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    man = read_json(os.path.join(root, "hakken.json"), {}) or {}
    man.update({
        "schema_version": WORKSPACE_SCHEMA,
        "created_with": "hakken-discovery " + __version__,
        "created_at": man.get("created_at") or now_iso(),
        "goal": goal or man.get("goal", ""),
        "domain": domain or man.get("domain", ""),
        "time_unit": time_unit,
        "git_policy": git_policy,
        "policy": {
            # pair-level searches reveal unpublished hypotheses to third parties
            "external_novelty_search": external_novelty_search,   # ask | allow | deny
            # single-entity literature lookups for the language branch
            "entity_evidence_search": entity_evidence_search,     # allow | ask | deny
        },
    })
    write_json(os.path.join(root, "hakken.json"), man)
    write_text(os.path.join(root, ".gitignore"), GITIGNORE[git_policy])
    write_text(os.path.join(root, "README.md"), README.format(version=__version__, policy=git_policy))
    for name, default in (("relations.json", DEFAULT_RELATIONS), ("entities.json", DEFAULT_ENTITIES)):
        fp = os.path.join(root, "config", name)
        if not os.path.exists(fp):
            write_json(fp, default)
    seeds = os.path.join(root, "config", "seeds.txt")
    if not os.path.exists(seeds):
        write_text(seeds, "# one focus entity per line (optional)\n")
    log("[init] workspace ready at %s (git policy: %s)" % (root, git_policy))
    return Workspace(root)
