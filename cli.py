import argparse
import json
from pathlib import Path

from rag_eval import RetrievalCase, evaluate, lexical_groundedness


def _string_list(value: object, field: str, limit: int) -> list[str]:
    if not isinstance(value, list) or len(value) > limit or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a bounded list of strings")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a bounded, recorded retrieval dataset.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("input must be a JSON object")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases or len(raw_cases) > 10_000:
        raise SystemExit("cases must be a non-empty bounded list")

    cases: list[RetrievalCase] = []
    try:
        for raw in raw_cases:
            if not isinstance(raw, dict) or not isinstance(raw.get("query"), str):
                raise ValueError("each case requires a text query")
            relevant = _string_list(raw.get("relevant_ids", []), "relevant_ids", 1_000)
            retrieved = _string_list(raw.get("retrieved_ids", []), "retrieved_ids", 1_000)
            cases.append(RetrievalCase(raw["query"][:2_000], frozenset(relevant), tuple(retrieved)))
        output = evaluate(cases, args.k)
        if "answer" in data or "contexts" in data:
            if not isinstance(data.get("answer"), str):
                raise ValueError("answer must be text")
            contexts = _string_list(data.get("contexts"), "contexts", 100)
            output["lexical_groundedness_proxy"] = lexical_groundedness(
                data["answer"][:20_000], [context[:20_000] for context in contexts]
            )
    except (TypeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
