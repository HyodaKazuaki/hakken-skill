# Report template

`H report` generates this layout with sections 1–4 filled from the run; complete every
`<!-- agent -->` placeholder. Write the final report in the user's language. Keep scores, triples and years exact.

```markdown
# Hakken discovery report: <topic>

## 1. Setup
- Goal: <research question>
- Graph: <N entities>, <M triples>, <R relation types>, years <a–b>, sources <…>
- Knowledge cutoff: <year>; seeds: <n> (<how chosen>)
- Graph branch: temporal-network-feature logistic model fitted on (T−W, T] (or heuristic)
  [+ language-branch fusion α=<…>, judge model <…>]. Not the THiGERLLM neural model.

## 2. Does the graph predict the future? (back-test)
| cutoff | bin | micro recall | null | macro recall | null | nDCG | covered |
|---|---|---|---|---|---|---|---|
Interpretation in 2–3 sentences. State the prediction budget (--top-frac).

## 3. Selection funnel
| step | hypotheses |
|---|---|
| scored | … |
| novel (pair unconnected) | … |
| top-k per entity | … |
| expert/agent filters | … |
| shortlisted | … |

## 4. Hypotheses
### H1. (<subject>, <RELATION>, <object>)
- Confidence: score <…> | prob <…> | relation percentile <…> | LLM plausibility <…>
- Novelty check: <searched queries, date> → no prior report / known since <year>
- Context: recency <subject year / object year>, shortest path <k>
- Evidence paths (PHELInE-style):
  1. <S> —REL→ <X> [year] —REL→ <O> [year]  (sufficiency <…>, necessity <…>)
  2. …
- Proposed mechanism: <one or two sentences>
- Why it could be wrong: <confounders, polarity uncertainty, context dependence>
- Minimal validation: <system/model>, <perturbation of S>, <readout of O>,
  <controls>, <expected direction>, <rough cost/time>
- Impact if true: <what it unlocks>

(repeat for H2…Hn)

## 5. Caveats
- Scores = literature support, not truth; unobserved ≠ false.
- Graph extraction misses negation/conditions; quantities not modelled.
- Possible LLM temporal leakage in the language branch.
- <data-specific caveats>

## 6. Reproduce
<exact commands, config files and data paths>
```

## Hypothesis card (short form, for tables or chats)
`H# | (S, R, O) | score/prob | top path | novelty | validation idea`
