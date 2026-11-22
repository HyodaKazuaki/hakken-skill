# Data pipeline: from literature to a temporal knowledge graph

## Contents
- [Target schema](#target-schema)
- [Sources](#sources)
- [Agent extraction protocol](#agent-extraction-protocol)
- [Clean config format](#clean-config-format)
- [Quality checks](#quality-checks)

## Target schema
Observations go into the workspace ledger (`hakken/ledger/observations.csv`, see
`workspace.md`). To add your own, write CSV/TSV with one row per observation (a source
stating a fact); repeats across sources are wanted (they become the `support` feature):

```
subject,relation,object,year,source,evidence,subject_type,object_type
TP53,AFFECTS_EXPRESSION_OF,BAMBI,2011,PMID:123456,"p53 induction increased BAMBI ...",gene,gene
```
then `H ingest file.csv --extractor user|agent:<model-name>|<database>`.
Required: `subject, relation, object, year` (aliases head/predicate/tail/date accepted).
Use canonical identifiers or preferred names consistently (HGNC symbols, ChEBI names, MeSH
terms, material formulas…). Relations: UPPER_SNAKE, directed, active voice.

## Built-in fetcher (biomedicine)
`H fetch-pubtator --query @GENE_SIRT1 --query "cellular senescence" --year-from 1995 --year-to 2025 --per-year 30`
- Queries are PubTator3 entity ids (`@GENE_<symbol>`, `@DISEASE_<Name>`, `@CHEMICAL_<Name>`)
  or free text. Per-year stratification is essential: plain relevance search returns mostly
  recent papers and leaves no history to learn from.
- Each relation in each document becomes one observation (`source=PMID:..`,
  `extractor=pubtator3`). PubTator3 relation types: ASSOCIATION, POSITIVE_CORRELATION,
  NEGATIVE_CORRELATION, BIND, COTREATMENT, COMPARISON, DRUG_INTERACTION, … — see
  `examples/pubtator_relations.json` for a ready config (ASSOCIATION as polarity parent).
- Rate-limited (~3 requests/s) and cached in `~/.cache/hakken/http`; ~1,000 documents per
  minute when cached, a few minutes uncached.

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

## Relations and entities config
Cleaning is not a separate step: `config/relations.json` and `config/entities.json` are
applied every time a command builds the graph from the ledger, so the ledger stays untouched.

`config/relations.json`:
```json
{
  "definitions": {"TREATS": "subject is used to treat or cure object (disease)",
                  "INCREASES_EXPRESSION_OF": "subject raises mRNA/protein level of object"},
  "rename":   {"USED_FOR_TREATMENT_OF": "TREATS", "OF": "IS_A_BIOMARKER_FOR"},
  "swap":     {"IS_TREATED_BY": "TREATS", "EXPRESSION_IS_INCREASED_IN": "INCREASES_EXPRESSION_OF"},
  "drop":     ["CO_OCCURS_WITH"],
  "keep_only": [],
  "polarity": [{"parent": "AFFECTS_EXPRESSION_OF",
                "children": ["INCREASES_EXPRESSION_OF", "DECREASES_EXPRESSION_OF"]}],
  "upper_snake": true,
  "min_relation_count": 1,
  "fusion_alpha": {"IS_A_BIOMARKER_FOR": 0.4}
}
```
`config/entities.json`:
```json
{"synonyms": {"p53": "TP53", "Tylenol": "acetaminophen"},
 "exclude": ["Neoplasms", "cell"],
 "lowercase": false}
```
- `swap` reverses subject/object and renames (passive → active).
- `polarity` (paper Table E2): child only → parent added with the child's year; both children
  → both removed, parent kept with the earliest year.
- `exclude` keeps generic hubs out of shortlists (they still inform the graph).
- `lowercase` only when names are not case-sensitive (not for gene symbols).
- `definitions` are shown to language-branch judges — write them carefully.
- Ready examples: `examples/biomedical_relations.json` (paper Table E2/E3),
  `examples/pubtator_relations.json`.
- One-off files outside a workspace: `H stats --triples f.csv --config cfg.json`.

## Quality checks
Run `H stats` after each config change and look for:
- Year histogram: gaps or a spike in one year (bulk import artefact) distort temporal
  signals; restrict to a sensible range.
- Hubs: generic nodes with huge degree (e.g. "cancer", "cell", "temperature") —
  merge to specific concepts or exclude from selection.
- Relation balance: a single dominant relation makes macro metrics essential.
- Isolated seeds: seed entities missing from the graph cannot be predicted for.
