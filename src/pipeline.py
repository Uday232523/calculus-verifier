import os
import re
from dotenv import load_dotenv
from sympy import sympify, simplify
from openai import OpenAI

load_dotenv()

# Point to Hugging Face's New Router API (much more stable)
client = OpenAI(
    base_url="https://router.huggingface.co/v1", api_key=os.environ.get("HF_TOKEN")
)

# Use the 7B model. It is always awake, lightning fast,

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def extract_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def verify_answer(llm_output, ground_truth_str):
    try:
        return simplify(sympify(llm_output) - sympify(ground_truth_str)) == 0
    except Exception as e:
        print(f"  [SymPy Parse Error: {e}]")
        return False


def evaluate_problem(problem_expr, ground_truth_str):
    print(f"\n--- Testing Problem: d/dx [ {problem_expr} ] ---")

    system_instruction = (
        "You are a strict calculus engine. You MUST output your final mathematical "
        "expression enclosed inside <answer> and </answer> tags. "
        "Use plain Python/SymPy syntax inside the tags (e.g., x**2, a/b, *). Do NOT use LaTeX."
    )

    conversation = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": f"Find d/dx [ {problem_expr} ]. Show work, then put the final expression inside <answer> tags.",
        },
    ]

    print(f"🤖 Sending Round 1 to Hosted {MODEL_NAME}...")
    response1 = client.chat.completions.create(
        model=MODEL_NAME, messages=conversation, temperature=0.7
    )

    reply1 = response1.choices[0].message.content
    answer1 = extract_answer(reply1)

    if answer1 is None:
        print("⚠️ Model forgot the tags. Parsing raw output...")
        answer1 = reply1.strip().split("\n")[-1]

    print(f"Model Answer 1: {answer1}")

    if verify_answer(answer1, ground_truth_str):
        print("✅ Correct on First Try!")
        return "correct_baseline", answer1

    # THE FEEDBACK LOOP
    print("❌ Incorrect. Triggering Feedback Loop...")

    conversation.append({"role": "assistant", "content": reply1})
    conversation.append(
        {
            "role": "user",
            "content": "Your answer failed mathematical verification. Please carefully review your steps and provide a corrected Python/SymPy format expression inside <answer> tags.",
        }
    )

    print("🤖 Sending Round 2 (Reprompt)...")
    response2 = client.chat.completions.create(
        model=MODEL_NAME, messages=conversation, temperature=0.0
    )

    reply2 = response2.choices[0].message.content
    answer2 = extract_answer(reply2)

    print(f"Model Answer 2: {answer2}")
    if answer2 and verify_answer(answer2, ground_truth_str):
        print("✅ Successfully self-corrected!")
        return "correct_feedback", answer2
    else:
        print("❌ Still incorrect.")
        return "failed", answer2
