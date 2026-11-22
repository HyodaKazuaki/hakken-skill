---
name: hakken-discovery
description: >-
  Hakken-style knowledge prediction for any research field. Builds a temporal
  knowledge graph of (subject, relation, object, year) facts from the literature,
  predicts as-yet-unpublished relationships between concepts, back-tests the
  predictions on historic data against a null baseline, explains each hypothesis
  with sufficiency/necessity-scored evidence paths (PHELInE-style), and filters
  them into a short list for expert review and experimental validation. Use when
  the user wants to generate, prioritise, or evaluate novel research hypotheses,
  do literature-based discovery, find missing links or "what will be discovered
  next" in a field, drug repurposing / target discovery, materials or agricultural
  idea generation, or asks for a Hakken / THiGER / PHELInE workflow.
  日本語: 研究仮説の生成・発見予測・文献ベース発見・知識グラフによる未知の関係予測。
license: MIT
metadata:
  version: "1.0.0"
  based_on: "Besold et al., Hakken: Predicting future discoveries to fill the gaps in today's knowledge, arXiv:2609.04494 (2026)"
---

# Hakken Discovery

Turn a research question into **ranked, explained, testable hypotheses** of the form
`(subject, relation, object)` by learning how knowledge in a field has grown over time.
Follows the Hakken system (arXiv:2609.04494): temporal knowledge graph → relation
prediction → calibrated confidence → back-testing → path explanations → selection →
expert review → validation design.

Core idea to keep in mind: a Hakken confidence score is **the weight of literature
evidence that a relation should already exist given everything else known**, not proof
it is true. Unobserved facts are *unlabeled*, not false (positive–unlabeled setting).

## When to use / not use

- Use: hypothesis generation, "what is missing between A and B", literature-based
  discovery, drug repurposing/target ID, prioritising candidates, testing a user's own
  hypothesis against the literature, comparing fields, research-gap mapping.
- Do not use for: plain literature summaries, single fact lookups, or when the user
  only wants a review article (just answer/search directly).

## Toolkit

`scripts/hakken_kg.py` (Python 3.8+, standard library only — no install needed).
Resolve the path relative to this SKILL.md. Input is a CSV/TSV with columns
`subject, relation, object, year` (aliases like head/predicate/tail/date accepted).

| Command | Paper step | Purpose |
|---|---|---|
| `stats FILE` | §4.1 | size, years, relation and hub distribution |
| `clean FILE --config cfg.json` | App. E | synonyms, rename/swap/merge relations, earliest occurrence, polarity conflicts, rare relations |
| `predict FILE --cutoff Y [--calibrate W]` | §2, App. B | score unobserved `(s,r,o)`; optional Platt calibration on window `(Y-W, Y]` |
| `backtest FILE --cutoff Y` | §4.3 | interval recall vs. random-empirical-score null, macro recall, label nDCG per future bin |
| `explain FILE -s S -r R -O O` | §3, App. D | path explanations with sufficiency / necessity, diversity rerank, `--latent` paths |
| `select PRED --triples FILE` | App. F.2 | novelty filter, top-k per entity, recency, shortest-path length |

Run `python3 scripts/hakken_kg.py <cmd> -h` for all options. A demo dataset is in
`examples/toy_triples.csv` (planted mechanisms; backtest should beat the null clearly).

The built-in scorer is a transparent surrogate (temporal evidence accumulation +
2/3-hop path structure + mined relation-pattern rules + node relation profiles), not
the THiGERLLM neural model. Say so in reports. For the full model the paper points to
https://github.com/SonyResearch/HakkenOSS. You (the agent) supply the paper's
**language branch**: semantic plausibility judgements grounded in pre-cutoff text.

## Workflow

Copy this checklist into your working notes and tick it off:

```
- [ ] 1 Frame   : goal, domain, entity types, relation vocabulary, time unit, seed set
- [ ] 2 Build   : temporal KG triples with first-publication year
- [ ] 3 Clean   : normalise, reconcile polarity, earliest occurrence, prune
- [ ] 4 Validate: backtest at 1-3 historic cutoffs; model must beat null
- [ ] 5 Predict : score candidates at the current cutoff (calibrate if data allows)
- [ ] 6 Fuse    : add LLM semantic assessment (language branch) -> ensemble
- [ ] 7 Select  : novelty, top-k per entity, recency, path length, expert filters
- [ ] 8 Explain : PHELInE-style paths for every shortlisted hypothesis
- [ ] 9 Report  : ranked hypotheses + evidence + validation designs + caveats
```

### 1. Frame the discovery problem
Pin down with the user (ask only what you cannot infer):
- **Research goal / topic** (e.g. aging, perovskite stability, soil microbiome).
- **Entity types** (gene, drug, disease / material, property, process / crop, practice…).
- **Relation vocabulary**: 5–30 typed, directed relations. Prefer mechanistic verbs
  (`INCREASES_EXPRESSION_OF`, `INHIBITS`, `ENHANCES_PROPERTY`) over `RELATED_TO`.
  Define a parent for opposing pairs (e.g. `AFFECTS` over `INCREASES`/`DECREASES`).
- **Time unit** (publication year by default) and the **knowledge cutoff**.
- **Seed entities** to focus on (the paper used 1,386 expert-chosen aging genes).
See `references/domain-adaptation.md` for ready-made schemas per field.

### 2. Build the temporal knowledge graph
Priority order: (a) user-provided triples/database; (b) structured sources with
dates (PubTator3, SemMedDB, OpenAlex, Crossref, domain DBs); (c) extraction you do
from abstracts/full text with search tools. Every triple needs the **year it was first
published**. Extraction rules and source list: `references/data-pipeline.md`.
Aim for thousands of triples at minimum; the method relies on structure accumulating
over time. With < ~300 triples, say the predictions are exploratory only.

### 3. Clean
Write a `clean` config (synonyms, renames, swaps for passive relations, polarity groups,
drop list), run `clean`, then `stats`. Check hubs: generic nodes ("cell", "protein",
"disease") create trivial hypotheses — merge, drop or list them for `--exclude-entities`.

### 4. Validate on history before trusting anything
```
python3 scripts/hakken_kg.py backtest data.csv --cutoff <Y-10> --bin 2 --bins 5
```
Repeat for 2–3 cutoffs. Report micro/macro recall per bin **versus the null**, nDCG,
and `covered`. Do not report precision as a quality measure (unknown ≠ false).
If the model does not clearly beat the null, the graph is too sparse or noisy — fix
data (step 2–3) or tune `--hops 3`, `--tau`, `--hub-cap` before predicting.
Interpretation guide: `references/evaluation.md`.

### 5. Predict
```
python3 scripts/hakken_kg.py predict data.csv --cutoff <latest> --calibrate 5 \
    [--seeds seeds.txt --seed-mode both] [--hops 3] -o predictions.csv
```
`prob` (when calibrated) ≈ chance the relation gets reported within W time units,
a lower bound under PU. Without calibration use `rel_percentile` to compare across
relation types. Only use `--hops 3` / `--seed-mode both` when the graph or seed set
is small enough (cost grows with pairs × neighbours²).

### 6. Fuse the language branch (agent reasoning)
For the top ~50–200 hypotheses, judge semantic plausibility *as the paper's text
branch would*: from what is known about each entity **separately** before the cutoff.
- Retrieve or recall short descriptions of `s` and `o` individually; do not search for
  the pair itself (that would leak the answer and destroy novelty).
- Score 0–1: is there a coherent mechanism? Is the relation type and direction right?
- Combine label-wise: `final = α·graph_rank + (1-α)·llm_score` (α≈0.5; raise α for
  dense graphs, lower it for rare relations where text helps most).
- Then, and only then, search the literature for the pair: if already published,
  mark **"known (missed by graph)"** and drop it from the novel list.
Protocol and prompt template: `references/method.md#language-branch`.

### 7. Select
```
python3 scripts/hakken_kg.py select predictions.csv --triples data.csv \
    --top-k 3 --score-col prob --min-score 0.5 --report selection.md -o shortlist.csv
```
Then apply expert criteria with the user: relevance to the goal, recency (hot vs.
nascent area), path length (longer = less obvious, riskier), testability, cost.

### 8. Explain every shortlisted hypothesis
```
python3 scripts/hakken_kg.py explain data.csv -s S -r R -O O --max-len 3 --latent
```
Translate the top diverse paths into mechanism sentences ("S regulates X [2004],
and X increases expression of O [2011] → plausible S → O"). High sufficiency = the
path alone supports it; high necessity = the prediction collapses without it.
A hypothesis with no paths is speculative — say so.

### 9. Report and propose validation
Use `references/report-template.md`. For each hypothesis give: triple, score/prob,
evidence paths with years, LLM rationale, novelty check result, risks/alternative
explanations, and a **minimal experiment** (perturb S, measure O; controls; readout;
expected direction). In the paper, 2 of 3 top hypotheses passed wet-lab tests —
communicate that a confirmed fraction well below 100% is normal.

## Guardrails
- **Temporal hygiene**: build the graph only from data ≤ cutoff; when evaluating,
  never let post-cutoff text inform scores. Warn that LLM pretraining may leak future
  knowledge (the paper notes this too).
- **Novelty**: a hypothesis is novel only if the pair has no recorded relation in the
  graph *and* a targeted literature search finds no report.
- **Polarity & negation**: extraction misses negated/speculative statements; treat
  `INCREASES` vs `DECREASES` predictions as direction-uncertain unless paths agree.
- **Hubs** dominate raw scores; always use top-k per entity and inspect hubs.
- **Never present scores as probabilities of truth**; present them as literature
  support. Keep quantitative/conditional limits (dose, environment) explicit.
- Small or synthetic graphs: label results as illustrative.

## References (load on demand)
- `references/method.md` — paper summary: THiGERLLM, PHELInE, losses, calibration, what the surrogate keeps/drops, language-branch protocol.
- `references/data-pipeline.md` — sources, extraction prompt, schema, cleaning config format.
- `references/domain-adaptation.md` — relation schemas and sources for biomedicine, materials, chemistry, agriculture, ecology, social science, CS/AI, physics.
- `references/evaluation.md` — temporal split protocol, metrics, null baseline, reading results.
- `references/report-template.md` — hypothesis card and final report layout.
