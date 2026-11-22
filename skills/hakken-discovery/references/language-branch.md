# Language branch: the agent as THiGERLLM's text model

In Hakken, a fine-tuned LLM reads ~20 sentences about the subject **alone** and ~20 about the
object **alone** (all published before the cutoff), plus a compact summary of the graph, and
scores each relation type. Its scores are combined label-wise with the graph branch
(`ℓ = α·ℓ_graph + (1−α)·ℓ_text`) and calibrated. Text helps most for rare relations and sparse
graph regions (paper: macro recall 46 → 61%). Here the agent (or its sub-agents) plays that role.

## Contents
- [Pipeline](#pipeline)
- [Roles and information firewall](#roles-and-information-firewall)
- [Orchestration by agent capability](#orchestration-by-agent-capability)
- [Judge protocol](#judge-protocol)
- [Fusion](#fusion)
- [Known limitations](#known-limitations)

## Pipeline
```
H judge-queue --top 50 --batch-size 10   -> runs/<id>/judge/batch_###.jsonl + PROMPT.md
(sub-)agents judge each batch            -> judgments_###.jsonl (one JSON object per item)
H record judgments judgments_###.jsonl   -> hakken/judgments/llm.jsonl (validated, append-only)
H fuse [--alpha 0.5]                     -> runs/<id>/fused.csv
H select                                 -> uses fused.csv automatically
```
Each queue item contains: id, subject, relation, object, cutoff, graph_score, relation
definition, alternative labels for the pair, up to 3 graph evidence paths (with years), and
entity types. Already-judged items (same triple, cutoff, prompt version) are skipped, so the
queue can be re-run safely after interruptions.

## Roles and information firewall
| Role | Sees | Must not |
|---|---|---|
| Orchestrator (main agent) | everything | send pair queries without policy approval |
| Evidence collector (optional separate sub-agent) | one entity name + cutoff | search the two entities together |
| Judge | queue item + evidence for each entity | search the pair; use post-cutoff knowledge |
| Red-team (optional) | judged item | change scores; it adds alternative explanations to `risk` |
| Novelty checker | pairs, after judging | run before judgments exist (would bias judges) |

Why: if a judge sees literature mentioning both entities, it simply reads off the answer;
the prediction then measures retrieval, not plausibility, and temporal back-tests become
meaningless.

Evidence command (single entity, pre-cutoff, sentences mentioning the other entity removed):
```
H evidence --entity "SIRT1" --before 2015 --exclude-term "FOXO3" --context "gene OR protein" --source pubmed --max 20
```
Use `--context` with the entity type to avoid homonyms (e.g. "BAMBI" is also an autism
questionnaire). Never put the other entity into `--context`.
Outputs JSON with `sentences[{id, year, sentence}]`; cite the ids in the judgment's `evidence`.
If the network or policy forbids it, use only the agent's own knowledge **as of the cutoff**
and state that in `risk`.

## Orchestration by agent capability
- **Agents with parallel sub-agents** (e.g. Claude Code's agent/task tool, other harnesses with
  sub-agent spawning): spawn one judge per batch file. Give each the absolute paths of
  `PROMPT.md`, its batch file and an output path `judge/out_###.jsonl`; tell it to return only
  when the file is written. Keep 3–8 in flight. Then `record` every output file.
- **Agents without sub-agents**: process batches sequentially in the main context; write each
  batch's output file before starting the next (so work survives interruptions).
- **Headless / CI**: same as sequential; never block on questions.
- Use a capable model for judging; record its name in `model`. Mixing models is fine — they are
  recorded per judgment.
- Variance control (optional): judge the top items twice with different evidence samples and
  record both; `fuse` uses the latest — average them yourself into one record if you do this.

## Judge protocol
`PROMPT.md` in each judge directory is the canonical version (prompt_version `judge-v1`).
Summary:
1. Collect 5–20 statements about the subject alone and the object alone (≤ cutoff).
2. Read graph evidence paths and alternative labels.
3. Answer: coherent mechanism? correct relation type and direction?
4. Emit JSON: `id, subject, relation, object, cutoff, score (0–1), best_relation, mechanism,
   risk, evidence[], leak_suspected, model, prompt_version`.

Score anchors: 0.1 no plausible mechanism · 0.3 weak/indirect · 0.5 plausible but generic ·
0.7 specific mechanism supported by both entities' evidence · 0.9 strong, specific, consistent
with the graph paths.

`leak_suspected: true` when the judge recognises the pair as already published (common with
large LLMs for pre-training-era cutoffs). `fuse` drops these by default (`--keep-leaky` to keep).

## Fusion
`fuse` combines per hypothesis: `fused = σ(α·logit(p_graph) + (1−α)·logit(p_llm))`.
- `p_graph` = calibrated `prob` (features model) or `rel_percentile` as a proxy.
- `α` default 0.5; per-relation overrides in `relations.json` → `fusion_alpha`. Lower α for rare
  relation types and sparse regions; raise it for dense, well-studied areas.
- Unjudged hypotheses keep `fused = p_graph` and `judged = 0`; `--judged-first` sorts judged
  ones to the top.
- α is not fitted automatically: a fair fit would require judging historic hypotheses with an
  LLM that has not read the later literature. If the user wants an empirical α, judge a
  back-test run whose cutoff is after the judge model's training cutoff.

## Known limitations
- LLM judges may know post-cutoff literature (temporal leakage); only predictions evaluated
  after the model's training cutoff are clean.
- Judgments are non-deterministic and cost tokens; keep queues to the top 50–500.
- The judge does not see the full graph; it sees the top paths only. Graph structure remains
  the graph branch's job.
