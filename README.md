# RAG Eval Workbench

RAG Eval Workbench is a small local evaluation project for studying retrieval quality without having to keep a full RAG stack running. It works with recorded JSON fixtures, calculates common ranking metrics, and can also report a simple lexical groundedness proxy when an answer and its contexts are included.

I built it to make retrieval experiments easier to inspect. Instead of hiding the evaluation behind a framework, the metric functions are kept small enough to read, test and reason about directly.

## What it measures

- **Precision@K** — how much of the retrieved top-K set is relevant
- **Recall@K** — how much of the known relevant set was recovered
- **MRR** — how early the first relevant result appears
- **nDCG@K** — ranking quality for binary relevance labels
- **Lexical groundedness proxy** — token overlap between an answer and supplied contexts

Duplicate document IDs are counted only once inside the evaluated ranking. The groundedness value is deliberately described as a proxy: it is useful for a quick lexical check, but it is not a factuality or semantic-correctness score.

## Project structure

- `rag_eval.py` contains the metric functions and retrieval case model
- `cli.py` validates bounded local JSON input and prints deterministic JSON output
- `examples/` contains synthetic recorded fixtures
- `tests/` covers metric behavior, duplicate retrievals, invalid `k` values and groundedness edge cases
- `.github/workflows/tests.yml` runs the unit test suite on Python 3.12

The project has no runtime dependency on an LLM provider, vector database or external URL.

## Run locally

Python 3.12 or newer is sufficient.

```bash
python cli.py examples/sample.json --k 3
```

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## Engineering notes

Input size, query length, context count and retrieved/relevant identifier lists are bounded at the CLI layer. The core metric functions do not perform I/O, which keeps repeated runs deterministic and makes the calculations easy to test independently.

## Limitations

The evaluator assumes that relevance labels are already available. nDCG currently uses binary relevance rather than graded judgments. The lexical groundedness proxy only measures token overlap, so paraphrases and semantically equivalent wording can score lower than expected.

This repository is an evaluation lab rather than a complete production RAG observability platform. There is no live ingestion, model execution, vector search or automatic labeling.

## Possible next steps

Useful extensions would include graded relevance, confidence intervals, comparison reports across datasets and optional offline embedding-based analysis with clearly documented provenance.

## License

See [LICENSE](LICENSE).
