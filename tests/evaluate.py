import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qa_engine import ContextQA

def main():
    engine = ContextQA.from_directory(ROOT / "knowledge_base")
    cases = json.loads((ROOT / "tests" / "questions.json").read_text(encoding="utf-8"))
    results = []
    correct = 0
    tp = fp = tn = fn = 0
    for case in cases:
        result = engine.answer(case["question"])
        predicted = result["answered"]
        expected = case["answerable"]
        is_correct = predicted == expected
        correct += int(is_correct)
        if predicted and expected: tp += 1
        elif predicted and not expected: fp += 1
        elif not predicted and not expected: tn += 1
        else: fn += 1
        results.append({
            "id": case["id"], "question": case["question"],
            "expected_answerable": expected, "predicted_answerable": predicted,
            "correct": is_correct,
            "top_source": result["sources"][0]["file"] if result["sources"] else None,
            "answer": result["answer"]
        })
    total = len(cases)
    summary = {
        "total_questions": total,
        "correct": correct,
        "answerability_accuracy": round(correct / total, 4) if total else 0,
        "true_positive_answerable": tp, "false_positive_unanswerable": fp,
        "true_negative_unanswerable": tn, "false_negative_answerable": fn,
        "abstention_precision": round(tn / (tn + fn), 4) if (tn + fn) else 0,
        "abstention_recall": round(tn / (tn + fp), 4) if (tn + fp) else 0,
        "results": results
    }
    out = ROOT / "tests" / "evaluation_results.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f'Answerability accuracy: {summary["answerability_accuracy"]:.1%} ({correct}/{total})')
    print(f'Unanswerable correctly rejected: {tn}; unsupported answers: {fp}')
    print(f'Answerable correctly answered: {tp}; answerable missed: {fn}')
    print("\nPer-question outcomes:")
    for row in results:
        status = "PASS" if row["correct"] else "FAIL"
        print(f'{status:4} | Q{row["id"]:02} | expected={str(row["expected_answerable"]):5} predicted={str(row["predicted_answerable"]):5} | {row["question"]}')
    print(f"\nFull results written to {out}")

if __name__ == "__main__":
    main()
