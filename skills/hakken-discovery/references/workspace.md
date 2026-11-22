# Workspace: where knowledge data lives

## Contents
- [Principles](#principles)
- [Layout](#layout)
- [Ledger schema](#ledger-schema)
- [Config files](#config-files)
- [Append-only logs](#append-only-logs)
- [Runs and reproducibility](#runs-and-reproducibility)
- [Git policies](#git-policies)
- [Operating modes and external queries](#operating-modes-and-external-queries)
- [Multiple agents and concurrency](#multiple-agents-and-concurrency)

## Principles
1. **Data belongs to the project, code belongs to the skill.** Skills are copied or symlinked
   per agent (`.claude/skills`, `.agents/skills`, …), overwritten on update and sometimes
   read-only. Nothing is ever written inside the skill directory.
2. **One discoverable location.** `hakken/hakken.json` marks the workspace; commands find it by
   walking up from the current directory (like git finds `.git`), or via `--workspace` /
   `$HAKKEN_WORKSPACE`. Every agent working in the project therefore shares the same data.
3. **Self-describing.** `hakken.json` carries a `schema_version`; `README.md` explains every
   file for humans and other agents.
4. **Source of truth vs. derived.** Expensive, non-reproducible inputs (literature observations,
   LLM judgments, novelty checks) are append-only logs. Everything else is derived into
   `runs/<id>/` and can be regenerated from the logs + config + the command in `manifest.json`.
5. **Scoped version control.** The workspace ships its own `.gitignore`; the project's root
   `.gitignore` is never edited.
6. **Safe writes.** Rewritten files are written to a temp file and atomically renamed; appends
   take a lock file (`*.lock`, stale after 10 minutes).
7. **Caches are disposable and global.** HTTP responses go to `$HAKKEN_CACHE` or
   `$XDG_CACHE_HOME/hakken` (default `~/.cache/hakken`), shared across projects.

## Layout
```
hakken/
├── hakken.json              schema_version, goal, domain, time_unit, git_policy, policy{...}
├── README.md                generated description of the layout
├── .gitignore               scoped rules for the chosen git policy
├── config/
│   ├── relations.json       vocabulary, definitions, rename/swap/drop/keep_only, polarity, fusion_alpha
│   ├── entities.json        synonyms, exclude (generic hubs), lowercase
│   └── seeds.txt            focus entities, one per line
├── ledger/observations.csv  SOURCE OF TRUTH (append-only)
├── judgments/llm.jsonl      language-branch judgments (append-only)
├── novelty/checks.jsonl     novelty checks incl. pending (append-only)
└── runs/<timestamp>_<kind>_<label>/
    ├── manifest.json        params, input hashes, skill version, command
    ├── model.json           fitted feature weights / calibration / external source
    ├── predictions.csv      graph-branch hypotheses
    ├── novelty_queue.jsonl  pairs awaiting an external check
    ├── judge/               batch_*.jsonl, PROMPT.md for (sub-)agents
    ├── fused.csv            graph + language branch
    ├── shortlist.csv, selection.md
    ├── explanations/        explanations.json / .md
    ├── backtest.json / .md  (backtest runs)
    └── report.md
```

## Ledger schema
`subject, relation, object, year, source, evidence, subject_type, object_type, extractor, ingested_at`
- One row per **observation** (a source stating a fact). The same triple from several sources
  appears several times; the graph keeps the earliest year and counts distinct sources
  (`support` feature) — this addresses the paper's "first occurrence only" limitation.
- `extractor`: `pubtator3`, `user`, `agent:<model-name>`, `<database>`. Agent extractions
  must say which model produced them.
- Rows are never edited. Corrections go into `config/` (synonyms, renames, drops), which is
  applied each time a graph is built. To retract bad rows, add their relation or entity to a
  `drop`/`exclude` rule, or (rarely, with the user's consent) rewrite the file.

## Config files
`relations.json` keys: `definitions` {REL: "one line"}, `rename` {OLD: NEW}, `swap` {PASSIVE:
ACTIVE} (reverses subject/object), `drop` [..], `keep_only` [..], `polarity` [{parent, children:
[pos, neg]}] (Hakken Table E2), `upper_snake` (normalise names), `min_relation_count`,
`fusion_alpha` {REL: α} (per-relation graph weight in `fuse`).
`entities.json` keys: `synonyms` {variant: canonical}, `exclude` [generic hubs: never shortlisted and
avoided as explanation intermediates],
`lowercase`.

## Append-only logs
- `judgments/llm.jsonl`: one object per judgment (`subject, relation, object, cutoff, score,
  best_relation, mechanism, risk, evidence[], leak_suspected, model, prompt_version, recorded_at`).
  `judge-queue` skips hypotheses already judged with the same cutoff and prompt version.
- `novelty/checks.jsonl`: `subject, relation, object, status (novel|known|co-mentioned|unclear|pending),
  method (ledger|external:pubmed|external:openalex|deferred|agent), query, hits, examples, note, cutoff`.
  The newest record per pair wins in `select` and `report`.

## Runs and reproducibility
`manifest.json` stores the parameters, the sha256 of the ledger and config files, the skill
version and the command. Re-running that command on the same ledger/config reproduces the
features model exactly (fixed random seeds). Runs are cheap; create new ones rather than
editing old ones. Delete old runs only when the user asks.

## Git policies
Chosen at `init --git-policy` (default `private`):

| Policy | Tracked | Ignored | When |
|---|---|---|---|
| `private` (default) | hakken.json, README, config/, ledger/ | runs/, judgments/, novelty/ | almost always: generated hypotheses are unpublished research ideas |
| `standard` | + judgments/, novelty/ | runs/ | private repositories where the team shares judgments |
| `none` | nothing | everything | scratch analyses |

The ledger contains only published literature facts and is safe to share; hypotheses
(predictions, judgments, novelty queries) are not. If the project repository is public, keep
`private` or `none`.

## Operating modes and external queries
`hakken.json` → `policy`:
- `entity_evidence_search` (`allow` default | `ask` | `deny`): single-entity lookups
  (`evidence`, `fetch-pubtator`).
- `external_novelty_search` (`ask` default | `allow` | `deny`): pair lookups (`novelty --external`).

Commands exit with code 3 when a query needs approval. The agent decides by mode:
- **Interactive**: show the queries, ask once per batch, then `--confirmed`.
- **Autonomous / unattended** (auto-approve, goal/loop runs, headless, CI, scheduled, sub-agents):
  never self-approve; use `novelty --mark-pending`, complete the rest of the workflow, and list
  pending checks in the report and final message.
- **Plan / read-only**: no writes or network; describe the commands.
- Changing a policy (editing `hakken.json` or `init --force --novelty-search allow`) only on
  the user's explicit request.

## Multiple agents and concurrency
Several agents (e.g. Claude Code and Codex, or parallel judge sub-agents) may use the same
workspace. Appends are locked; runs get unique directories; sub-agents should write their
judgments to separate files that the orchestrating agent registers with `record`.
