# Data pipeline: from literature to a temporal knowledge graph

## Contents
- [Target schema](#target-schema)
- [Sources](#sources)
- [Agent extraction protocol](#agent-extraction-protocol)
- [Clean config format](#clean-config-format)
- [Quality checks](#quality-checks)

## Target schema
CSV (or TSV), one row per observation; duplicates are fine (earliest year wins):

```
subject,relation,object,year,subject_type,object_type,source
TP53,AFFECTS_EXPRESSION_OF,BAMBI,2011,gene,gene,PMID:123456
```
Required: `subject, relation, object, year`. Use canonical identifiers or preferred
names consistently (HGNC symbols, ChEBI names, MeSH terms, material formulas…).
Relations: UPPER_SNAKE, directed, active voice (`TREATS`, not `IS_TREATED_BY`).

## Sources
Prefer sources that already give normalised entities + publication dates.

| Field | Source | Access | Notes |
|---|---|---|---|
| Biomedicine | PubTator3 (NCBI) | REST `https://www.ncbi.nlm.nih.gov/research/pubtator3-api/` | genes, chemicals, diseases, variants, species + relation types; the paper's open variant is trained on it |
| Biomedicine | SemMedDB | UMLS licence | SemRep predications with PMIDs/years |
| Biomedicine | PubMed E-utilities | `esearch`/`efetch` | abstracts + years for your own extraction |
| Any field | OpenAlex | `https://api.openalex.org/works?search=...` | works, years, concepts/topics, abstracts (inverted index) |
| Any field | Crossref, Semantic Scholar, arXiv API | REST | metadata, abstracts, years |
| Chemistry/materials | PubChem, ChEBI, Materials Project | REST | entity normalisation, properties |
| Agriculture | AGROVOC, CAB Abstracts, historical texts (Biodiversity Heritage Library) | REST / bulk | thesaurus for normalisation |
| Ecology | GloBI (species interactions), GBIF | REST | interactions with dates |

Respect rate limits and licences. Never send private user data in queries.

## Agent extraction protocol
When no structured triples exist, extract them yourself (this mirrors the paper's
rule-based extractor, App. E.1):

1. Define the relation vocabulary and entity types first (see `domain-adaptation.md`).
2. Retrieve documents for the topic **and for each seed entity**, across the whole time
   range (not only recent papers — the temporal signal needs history). Keep the
   publication year of each document.
3. Per sentence/abstract, extract triples with this prompt:

```
Extract factual relations stated as findings (not hypotheses, not negated, not
"may/might") from the text. Use only these relation types: {list with definitions}.
Map entities to canonical names: {normalisation hints}. Direction: subject acts on object.
Return JSON lines: {"subject":..,"relation":..,"object":..,"evidence":"<quote>"}
If nothing qualifies, return nothing.
Text ({year}, {doc id}): {text}
```
4. Attach `year` and `source` from document metadata (not from the text).
5. Spot-check ~30 triples against their evidence quotes; fix systematic errors in the
   prompt or relation definitions before scaling up.
6. Log negated/speculative statements separately if they matter (Hakken ignores them).

Scale guidance: tens of thousands of documents give meaningful graph structure;
hundreds give a toy. For large corpora, process in batches and dedupe incrementally.

## Clean config format
`python3 scripts/hakken_kg.py clean raw.csv --config clean.json --upper-relations -o data.csv`

```json
{
  "synonyms": {"p53": "TP53", "Tylenol": "acetaminophen"},
  "rename":   {"USED_FOR_TREATMENT_OF": "TREATS", "OF": "IS_A_BIOMARKER_FOR",
               "IS_METABOLITE_OF": "METABOLIZES"},
  "swap":     {"IS_TREATED_BY": "TREATS", "IS_METABOLIZED_BY": "METABOLIZES",
               "EXPRESSION_IS_INCREASED_IN": "INCREASES_EXPRESSION_OF"},
  "drop":     ["CO_OCCURS_WITH"],
  "keep_only": [],
  "polarity": [
    {"parent": "AFFECTS_EXPRESSION_OF",
     "children": ["INCREASES_EXPRESSION_OF", "DECREASES_EXPRESSION_OF"]},
    {"parent": "AFFECTS_ABUNDANCE_OF",
     "children": ["INCREASES_ABUNDANCE_OF", "DECREASES_ABUNDANCE_OF"]}
  ]
}
```
- `swap` reverses subject/object and renames (passive → active).
- `polarity` (paper Table E2): child only → add parent with the child's year; both
  children → delete both, keep the parent with the earliest year.
- `--min-relation-count N` removes extremely rare relations; `--lowercase` casefolds
  entity names (use only if names are not case-sensitive, e.g. not gene symbols).
- Rules are applied before `polarity`, so write polarity groups with final names.

## Quality checks
Run `stats` after cleaning and look for:
- Year histogram: gaps or a spike in one year (bulk import artefact) distort temporal
  signals; restrict to a sensible range.
- Hubs: generic nodes with huge degree (e.g. "cancer", "cell", "temperature") —
  merge to specific concepts or exclude from selection.
- Relation balance: a single dominant relation makes macro metrics essential.
- Isolated seeds: seed entities missing from the graph cannot be predicted for.
