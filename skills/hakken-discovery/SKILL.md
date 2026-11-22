---
name: hakken-discovery
description: >-
  Hakken-style knowledge prediction for any research field (arXiv:2609.04494).
  Builds a temporal knowledge graph of dated (subject, relation, object) facts from
  the literature in a project workspace, predicts as-yet-unpublished relationships
  with a fitted temporal-network-feature model (no GPU, no training infrastructure),
  adds an LLM "language branch" judged by the agent or its sub-agents, back-tests
  against a null baseline, explains hypotheses with sufficiency/necessity evidence
  paths (PHELInE-style), checks novelty, and shortlists hypotheses with validation
  designs. Use when the user wants novel research hypotheses, literature-based
  discovery, missing links or "what will be discovered next" in a field, drug
  repurposing / target discovery, materials or agricultural idea generation, or a
  Hakken / THiGER / PHELInE workflow.
  日本語: 研究仮説の生成・発見予測・文献ベース発見・知識グラフによる未知の関係予測。
license: MIT
compatibility: Python 3.8+ standard library only; network access optional (PubTator3, PubMed, OpenAlex).
metadata:
  version: "2.0.0"
  based_on: "Besold et al., Hakken: Predicting future discoveries to fill the gaps in today's knowledge, arXiv:2609.04494 (2026), https://arxiv.org/abs/2609.04494"
---

# Hakken Discovery

Turn a research question into **ranked, explained, testable hypotheses** `(subject, relation,
object)` by learning how knowledge in a field has grown over time. Mirrors Hakken:
temporal knowledge graph → graph-branch prediction → language branch → calibrated fusion →
back-test → path explanations → selection → validation design.

A score here is **the weight of literature evidence that a relation should exist given what was
known at the cutoff** — not proof. Unobserved relations are *unlabeled*, not false.

## Toolkit

`scripts/hakken.py` (Python 3.8+, stdlib only). Resolve the path relative to this file; below
`H` means `python3 <this skill dir>/scripts/hakken.py`. Every command has `-h`.

| Command | Hakken step | Purpose |
|---|---|---|
| `init`, `status` | — | create / inspect the project workspace |
| `fetch-pubtator`, `ingest` | §4.1, App. E | add dated observations (with provenance) to the ledger |
| `stats`, `export-graph` | App. E | inspect the cleaned graph (config applied) |
| `backtest` | §4.3 | interval recall vs. random-empirical-score null, macro recall, nDCG |
| `predict` | §2 | graph branch → new run with `predictions.csv` (`--import-scores` for external models) |
| `novelty` | App. F.2 | ledger check + optional external co-mention search (policy-gated) |
| `evidence` | §2 text branch | sentences about ONE entity published before the cutoff |
| `judge-queue`, `record`, `fuse` | §2, App. B.5 | language branch by (sub-)agents → logit ensemble |
| `select` | App. F.2 | novelty, top-k per entity, recency, path length → `shortlist.csv` |
| `explain` | §3, App. D | sufficiency / necessity path explanations, diversity rerank, latent paths |
| `report` | — | `report.md` skeleton with all evidence; you fill mechanism & experiment |

Graph branch (`--model features`, default): Science4Cast-style temporal network features
(common neighbours, Adamic–Adar, resource allocation, Jaccard, recency-weighted paths, degree
growth, node age, 3-hop paths, multi-source support) + relation-pattern/profile features,
combined by a logistic model **fitted in seconds on the graph's own history**: snapshot at
`T−W` → triples first published in `(T−W, T]`. Output `prob` ≈ chance of being reported within
`W` time units. `--model heuristic` uses fixed weights. This replaces THiGER's GNN; it is not
the THiGERLLM network — say so in reports. Method details: `references/method.md`.

## Where data lives

All knowledge data lives in a **project workspace** (`hakken/` found by walking up from the
CWD, or `--workspace` / `$HAKKEN_WORKSPACE`) — never in this skill directory. The ledger
(`ledger/observations.csv`) is the append-only source of truth; `runs/<id>/` holds derived,
reproducible outputs with a `manifest.json`. Download caches go to `~/.cache/hakken`.
Default git policy is `private` (track config + ledger only). Details, layout and git policies:
`references/workspace.md`.

## Operating-mode policy (external searches)

Two kinds of outbound queries exist; `hakken.json` holds the policy for each:
- `entity_evidence_search` (default `allow`): single-entity lookups (`evidence`, `fetch-pubtator`).
- `external_novelty_search` (default `ask`): searching a **pair** reveals an unpublished
  hypothesis to third parties.

Commands that send queries exit with code 3 unless the policy is `allow` or you pass
`--confirmed`. Decide by your current operating mode:

| Mode | What to do when the policy is `ask` |
|---|---|
| Interactive (user can answer now) | Show the exact queries, ask once per batch, re-run with `--confirmed` only after explicit approval |
| Autonomous / unattended (auto-approve, goal or loop runs, headless `-p`/`exec`, CI, scheduled, sub-agent) | Do not ask mid-run and do not send pair queries; run `novelty --mark-pending`, finish everything else, list pending checks in the report and final message |
| Plan / read-only | No writes, no network: describe the commands you would run |
| Unsure | Treat as autonomous |

Never pass `--confirmed` on your own judgement, never edit the policy unless the user asks,
and never let judge sub-agents search for pairs.

## Workflow

Copy this checklist into your notes and tick it off:

```
- [ ] 0 Workspace : H status (or H init --goal "...")
- [ ] 1 Frame     : relations.json (vocabulary + definitions), entities.json, seeds.txt
- [ ] 2 Ledger    : fetch-pubtator / ingest, with year + source + extractor
- [ ] 3 Inspect   : H stats ; fix synonyms, hubs, relation merges
- [ ] 4 Back-test : H backtest --cutoff <T-10>  -> model must beat null
- [ ] 5 Predict   : H predict  -> run id
- [ ] 6 Novelty   : H novelty --top 200 (+ external per mode policy)
- [ ] 7 Language  : H judge-queue -> (sub-)agents -> H record judgments -> H fuse
- [ ] 8 Select    : H select
- [ ] 9 Explain   : H explain
- [ ] 10 Report   : H report -> fill mechanism / risks / experiments -> present
```

**0–1 Frame.** Run `H status`; if no workspace exists, `H init --goal "<question>" --domain <d>`
(ask about the git policy only if the project is a public repository). Agree with the user on:
entity types, 5–30 directed relations in UPPER_SNAKE with one-line `definitions`, polarity groups
(e.g. `AFFECTS` over `INCREASES`/`DECREASES`), time unit, and seed entities. Write them to
`config/relations.json`, `config/entities.json`, `config/seeds.txt`. Schemas per field:
`references/domain-adaptation.md`.

**2 Ledger.** Every observation needs the year of first publication and a source id.
- Biomedicine: `H fetch-pubtator --query @GENE_TP53 --query "<topic>" --year-from 1995 --year-to 2025 --per-year 30`
  (always stratify by year; relevance-ranked fetching returns mostly recent papers).
- Other fields or user data: write a CSV `subject,relation,object,year,source,evidence` and
  `H ingest file.csv --extractor user|agent:<model-name>|<database>`. Extraction protocol:
  `references/data-pipeline.md`. Prefer thousands of observations spanning ≥15 years.

**3 Inspect.** `H stats`. Merge synonyms, add generic hubs (e.g. "Neoplasms", "cell") to
`entities.json` `exclude`, rename/swap/drop relations. Configuration is applied every time
the graph is built, so the ledger never needs rewriting.

**4 Back-test before trusting anything.** `H backtest --cutoff <latest−10>` (both models by
default; add a second cutoff if time allows). Report micro/macro recall per bin **vs. the
null**, nDCG and `covered`. Never report precision as quality. If the model does not clearly
beat the null, fix data or scope (seeds, `--hops 3`, `--hub-cap`) before predicting. Reading
guide: `references/evaluation.md`.

**5 Predict.** `H predict` (cutoff = latest year; `--window 5`; seeds from config;
`--seed-mode both` scores all seed pairs). Prints the run directory; keeps the top 50,000
(`--limit 0` = all). Cost is roughly 1 min per million candidate pairs; restrict with seeds or
`--hub-cap` on large graphs. External model scores: `H predict --import-scores scores.csv`.

**6 Novelty.** `H novelty --top 200` marks pairs already in the ledger as `known`. For the rest,
apply the operating-mode policy above (`--external pubmed|openalex --confirmed` or `--mark-pending`).
A co-mention is not proof of the relation — review the examples it returns.

**7 Language branch.** `H judge-queue --top 50` writes `runs/<id>/judge/batch_*.jsonl` and
`PROMPT.md`. Judge each batch yourself or with parallel sub-agents (one batch each), following
`references/language-branch.md`: evidence about each entity **separately and before the
cutoff** (`H evidence --entity X --before <cutoff> --exclude-term Y`), JSON output per item.
Then `H record judgments <file>` and `H fuse` (logit ensemble, `--alpha 0.5` = equal weight).

**8 Select.** `H select --top-k 3 --limit 30` → `shortlist.csv`, `selection.md`. Discuss with the
user: relevance, recency (hot vs. nascent), path length (longer = less obvious), testability.

**9 Explain.** `H explain` (whole shortlist) or `H explain -s S -r R -O O --latent`. High
sufficiency = the path alone supports the prediction; high necessity = it collapses without it.
No paths = speculative. Paths through `exclude` hubs are skipped unless nothing else exists
(`--allow-hubs` to show them).

**10 Report.** `H report` writes `runs/<id>/report.md`; fill every `<!-- agent -->` placeholder:
mechanism from the paths and judgments, why it could be wrong, and a **minimal experiment**
(system, perturbation of the subject, readout of the object, controls, expected direction).
Present the top hypotheses with the back-test context and any pending novelty checks. In the
paper 2 of 3 wet-lab tests confirmed — set expectations accordingly.

## Guardrails
- Temporal hygiene: features, judgments and evidence must use only data ≤ cutoff; flag
  `leak_suspected` when the LLM recognises a pair (those judgments are dropped by `fuse`).
- Novel = pair unconnected in the ledger **and** no report found by a novelty check.
- Polarity and negation are unreliable in extracted data; keep direction claims cautious.
- Hubs dominate raw scores: use `exclude` and top-k per entity.
- Never present scores as probabilities of truth; keep conditions/quantities explicit.
- Graphs < ~1,000 observations or < 10 years: label results exploratory.

## References (load when needed)
- `references/workspace.md` — workspace layout, ledger schema, git policies, run manifests, modes.
- `references/language-branch.md` — sub-agent orchestration, judge protocol, evidence rules.
- `references/method.md` — Hakken paper summary, what this implementation keeps / replaces.
- `references/data-pipeline.md` — sources, extraction prompt, relations/entities config format.
- `references/domain-adaptation.md` — relation schemas for biomedicine, materials, chemistry, agriculture, ecology, social science, CS/AI, physics.
- `references/evaluation.md` — back-test protocol, metrics, null baseline, calibration reading.
- `references/report-template.md` — final report layout and hypothesis card.
