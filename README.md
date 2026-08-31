# RAG Eval Workbench

## Overview
RAG Eval Workbench is a local, deterministic evaluator for bounded, recorded retrieval experiments. It reports ranking metrics and an explicitly named lexical-groundedness proxy without contacting a model provider, vector database, or remote URL.

## Problem
Retrieval experiments need repeatable metrics that can be inspected independently of a live RAG stack. A small recorded fixture makes metric behavior and limitations easy to review.

## Architecture
`cli.py` validates the JSON boundary and constructs immutable retrieval cases. `rag_eval.py` computes the metrics with no I/O. Tests exercise metric behavior and invalid inputs. Duplicate document identifiers receive credit only once within a ranking.

## Implemented Features
- Precision@K and Recall@K
- Mean Reciprocal Rank (MRR)
- nDCG@K for binary relevance labels
- Lexical token-overlap proxy for answer/context review
- Bounded local JSON input and deterministic JSON output

## Run Locally
Python 3.12 or newer is sufficient; there are no third-party dependencies.

```bash
python cli.py examples/sample.json --k 3
```

## Example
The included fixture evaluates one retrieval case and, when `answer` and `contexts` are present, adds `lexical_groundedness_proxy` to the output. The proxy measures token overlap only.

## Testing
```bash
python -m unittest discover -s tests -v
```

## Engineering Decisions
Metric functions are pure and deterministic. Input sizes, `k`, query length, context count, and retrieved/relevant identifiers are bounded at the CLI boundary. No transcript storage or network integration is required.

## Security / Privacy
Use recorded, non-sensitive fixtures. The tool does not fetch URLs, call an LLM, connect to a vector database, or transmit evaluation content.

## Limitations
Relevance labels are supplied by the evaluator. Binary nDCG does not model graded judgments. Lexical overlap misses semantic paraphrases and is not a factuality, correctness, or safety guarantee.

## Future Improvements
Potential extensions include graded relevance, confidence intervals, dataset comparisons, and optional offline embedding adapters with explicit provenance.

## License
MIT. See [LICENSE](LICENSE).
