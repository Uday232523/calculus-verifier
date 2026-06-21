import pandas as pd
import time
from pipeline import evaluate_problem


def run_evaluation(input_csv="data/problems.csv", output_csv="data/results.csv"):
    print(f"📂 Loading dataset from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Trackers for our metrics
    total_problems = len(df)
    correct_baseline = 0
    correct_feedback = 0
    failed = 0

    results = []

    for index, row in df.iterrows():
        problem_id = row["problem_id"]
        expr = row["expression"]
        ground_truth = row["ground_truth"]

        print(f"\n[{index + 1}/{total_problems}] Evaluating Problem {problem_id}...")

        # Call the pipeline for this specific problem
        status, final_model_answer = evaluate_problem(expr, ground_truth)

        if status == "correct_baseline":
            correct_baseline += 1
        elif status == "correct_feedback":
            correct_feedback += 1
        else:
            failed += 1

        # Save the result for this row
        results.append(
            {
                "problem_id": problem_id,
                "expression": expr,
                "ground_truth": ground_truth,
                "final_model_answer": final_model_answer,
                "status": status,
            }
        )

        # Save progress to CSV after every problem (so if it crashes, you don't lose data!)
        pd.DataFrame(results).to_csv(output_csv, index=False)

        # ⚠️ CRITICAL: Handle API Rate Limits
        # Google AI Studio free tier limits you to 15 Requests Per Minute (RPM).
        # Since one feedback loop takes up to 2 requests, we sleep to avoid getting banned.
        time.sleep(2)

    # --- Calculate and Print Final Metrics ---
    baseline_accuracy = (correct_baseline / total_problems) * 100
    total_correct = correct_baseline + correct_feedback
    post_feedback_accuracy = (total_correct / total_problems) * 100

    # Correction rate: out of the ones it got wrong initially, how many did it fix?
    initial_failures = total_problems - correct_baseline
    if initial_failures > 0:
        correction_rate = (correct_feedback / initial_failures) * 100
    else:
        correction_rate = 0

    print("\n" + "=" * 40)
    print("🏆 FINAL EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total Problems:           {total_problems}")
    print(f"Baseline (0-Shot) Score:  {correct_baseline} ({baseline_accuracy:.1f}%)")
    print(f"Fixed by Feedback Loop:   {correct_feedback}")
    print(f"Correction Rate:          {correction_rate:.1f}%")
    print(f"Final System Accuracy:    {total_correct} ({post_feedback_accuracy:.1f}%)")
    print("=" * 40)
    print(f"💾 Full results saved to {output_csv}")


if __name__ == "__main__":
    run_evaluation()
