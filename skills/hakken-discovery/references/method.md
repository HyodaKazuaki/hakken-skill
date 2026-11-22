# Method: Hakken in detail

Source: Besold, Akujuobi, Sanchez, et al. (Sony AI), *Hakken: Predicting future
discoveries to fill the gaps in today's knowledge*, arXiv:2609.04494v1, Sep 2026.

## Contents
- [Problem definition](#problem-definition)
- [THiGERLLM (prediction)](#thigerllm-prediction)
- [PHELInE (explanation)](#pheline-explanation)
- [Hypothesis selection](#hypothesis-selection)
- [Results worth quoting](#results-worth-quoting)
- [What this implementation keeps and replaces](#what-this-implementation-keeps-and-replaces)
- [Language branch protocol](#language-branch)
- [Limitations and future directions named by the authors](#limitations-and-future-directions)

## Problem definition
- Entities `V`, relation types `R = {1..C}`, triples `(s, r, o)`.
- Temporal KG: snapshots `G(1..T)`, each holding triples first observed up to time t.
- Task: for an (undirected) entity pair `(s, o)`, predict the **set** of relation types
  that hold → multi-label classification `y_so ∈ {0,1}^|R|`.
- Supervision is **positive–unlabeled**: `t_so,c > 0` gives the year label c was first
  observed; 0 means *unobserved*, not negative. A label-history snapshot
  `m(τ)_so,c = 1[0 < t_so,c ≤ τ]` tells the model which labels are already known at τ.
- "Knowledge prediction" vs. classic LBD: predictions may lie **outside the deductive
  closure** of current knowledge, specific in subject, object and relation type.

## THiGERLLM (prediction)
Two jointly trained branches + label history + calibration.

**Graph branch**
1. Node features = trainable embedding ⊕ ontology embedding (RDF2vec random walks over
   the ontology hierarchy → word2vec) ⊕ macro-domain embedding.
2. Per snapshot, GraphSAGE message passing with sampled neighbours → node states.
3. Pair embedding per snapshot `p(t) = W·Norm(h_s + h_o) + b` → sequence over T.
4. Hierarchical temporal Transformer: attend, then sum adjacent pairs, halving length
   each stage, until one global vector `z+`. A *pairmate mask* (−∞ attention between
   the two tokens about to be merged) forces cross-time information flow.
5. Graph logits `⟨W_z z+ + W_h h_hist + b, e_c⟩` against learned label embeddings;
   auxiliary link-ranking loss with negative pairs.

**Language branch** (Mistral-7B-Instruct-v0.3)
1. Text = 20 random sentences mentioning **s alone** and 20 mentioning **o alone**,
   all published before the cutoff (never sentences mentioning both).
2. Graph → LLM bridge: temporal latents downsampled to M prefix tokens (adaptive
   pooling, Fourier time encoding, σ-gate) + a global pair token spliced at `<htg>`.
3. Pool hidden states at entity markers `(h_E1 + h_E2)/2` → linear text logits +
   history-aware residual.

**Training & inference**
- Loss: PU-safe Dynamically-Weighted Focal BCE — focal factor, dynamic class weights,
  down-weighted unlabeled negatives, soft targets annealed by label priors, and a
  bounded confidence regulariser.
- Inference: label-wise ensemble `α_c·graph + (1−α_c)·text`, α tuned per label;
  blended Platt + isotonic calibration per label; per-label thresholds from
  precision–recall on validation.
- Output: each `(s, r, o)` with a calibrated confidence = estimated probability that the
  triple holds / weight of evidence that it "should already be in the literature".

## PHELInE (explanation)
Model-agnostic: only needs query access to the scorer `f(s, r, o)`.
1. **Candidate finder** — *corpus paths*: enumerate relational paths of length k
   between s and o in the observed graph (wet-lab run: shortest paths, 2–4 hops);
   *latent paths*: allow intermediate triples that are unobserved but scored highly
   by the model.
2. **Pseudo-retraining** — train a surrogate GNN (GraphSAGE) once to mimic
   `f`'s scores conditioned on the graph; then query it on modified graphs instead of
   retraining.
3. **Influence scoring** —
   *sufficiency(E)* = `f*(s,r,o | G_E)` (only the explanation facts present);
   *necessity(E)* = `f*(s,r,o | G) − f*(s,r,o | G \ E)` (drop when removed).
4. **Rerank** top-K for diversity (different intermediates / relation patterns) so the
   user sees several plausible mechanisms, not variants of one.

## Hypothesis selection
(App. F) For the aging study: 1,386 seed genes → 959,805 pairs → 1,543,297
above-threshold hypotheses →
1. *prepare*: uniform schema, drop pairs already related in the graph (1,524,046);
2. *recency*: median publication year per entity (hot vs. nascent);
3. *top-k per entity* (k=3) to stop hubs dominating (→ 2,804);
4. *path length*: shortest path in the pre-prediction graph (complexity proxy);
5. *finalise*; experts pick using explanations; 3 with confidence > 0.8 went to a CRO.

## Results worth quoting
- Data: 249M raw triples (PMC OA + MEDLINE + licensed) → 254,806 entities,
  7.1M triples, 23 relation types, 20 macro-domains.
- 2020 cut-off benchmark: THiGERLLM best macro-F1 (51.2) and macro-recall (60.7);
  THiGER (graph-only) slightly better weighted-F1/nDCG → text helps rare relations.
- Back-testing (cut-offs 1990/2000/2010, 2-year bins over 10 years): micro recall
  ≈0.71–0.75 in the first bin falling to ≈0.65–0.69, null ≈0.16–0.23; macro recall
  ≈0.50 → ≈0.44 (≈8% drop); nDCG ≈0.98 stable. Decay profile is the same across
  decades.
- Wet lab: TP53 → affects expression of → BAMBI (confirmed), RAF1 → decreases
  expression of → TNF (confirmed), SOAT1 → affects transcriptional activity of →
  STAT3 (not supported in the chosen assay).
- Also prototyped on 19th-century market-gardening texts (agriculture) and materials
  science.

## What this implementation keeps and replaces
| Hakken component | `hakken.py` |
|---|---|
| Temporal snapshots, first occurrence | ✔ cutoff snapshots from the ledger, earliest year kept; plus multi-source `support` |
| THiGER graph branch (GraphSAGE + temporal Transformer) | **replaced** by Science4Cast-style temporal network features (CN, Adamic–Adar, resource allocation, Jaccard, recency-weighted paths, degree growth, node age, 3-hop paths, support) + label features (mined path-pattern rules, node relation profiles, priors), one logistic model with per-relation intercepts fitted on `G(T−W) → (T−W, T]` in seconds (Newton–Raphson, stdlib) |
| PU-safe loss, class balancing | negatives subsampled with inverse-probability weights; probabilities = reported within W (lower bound) |
| Label history | `--include-connected` scores new relation types for already-connected pairs |
| Language branch (Mistral-7B) | **the agent / sub-agents** (`judge-queue` → `record` → `fuse`), entity-separated pre-cutoff evidence via `evidence` |
| Label-wise ensemble + calibration | `fuse`: `σ(α·logit p_graph + (1−α)·logit p_llm)`; graph branch is calibrated by construction (features) or Platt (heuristic) |
| PHELInE surrogate GNN | not needed: the feature model itself is re-queried on `G_E` / `G \ E` (exact pseudo-retraining) |
| Corpus / latent path candidates, diversity rerank | `explain` / `explain --latent` |
| Selection pipeline | `novelty` + `select` (prepare, recency, top-k per entity, path length) |
| Ontology embeddings (RDF2vec) | ✘ (use `synonyms`, entity types and `exclude` instead) |
| External / future models | `predict --import-scores` (e.g. HakkenOSS THiGER if weights become available) |

Why features instead of a GNN: in the Science4Cast benchmark (Krenn et al., arXiv:2210.00881),
"the most powerful methods use a carefully curated set of network features, rather than an
end-to-end AI approach" for predicting future links in an evolving concept network. HakkenOSS
(github.com/SonyResearch/HakkenOSS, BSD-3) contains THiGER code but no public weights and no
LLM branch, and training needs GPUs, PyG, ZenML and S3 — unsuitable for a portable skill.

## Language branch
See `language-branch.md` for the full protocol (roles, information firewall, sub-agent
orchestration, judge prompt, fusion).

## Limitations and future directions
Named by the authors — mention the relevant ones in reports:
- Only concepts and relations; no quantities or conditions (dose, temperature, cell line).
- Only the first occurrence of a fact; no count of supporting papers or recency of
  confirmation.
- Does not predict *when* a relation will be found, nor its downstream impact.
- Extraction ignores negation/speculation; polarity conflicts need expert rules.
- LLM pretraining cutoff can leak future knowledge.
- Suggested extension: generate an experimental design with each hypothesis (this skill
  does so in step 9).
- Complementary to LLM agents (AI Co-Scientist, ResearchAgent, HypoGeniC): Hakken
  scores can ground LLM-generated ideas in quantitative literature support.
