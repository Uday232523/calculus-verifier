# Multi-Stage Calculus Verification Engine with LLM Feedback Loop

An automated framework designed to evaluate, verify, and improve LLM mathematical reasoning performance on calculus problems (differentiation) using symbolic computation guardrails. 

This project implements a reliable **Feedback Loop Architecture** that catches LLM mathematical hallucinations and parsing issues in real-time, forcing the model to self-correct using deterministic mathematical verification.

---

## 🏗️ System Architecture

The project splits responsibility across three modular layers to maintain a clean separation of concerns:

1. **`src/generator.py`**: Automates the deterministic generation of the calculus evaluation dataset. It compiles standard functions, derives their ground-truth expressions using SymPy, and exports them cleanly.
2. **`src/pipeline.py`**: The core execution engine. Handles the stateful conversation with the LLM, injects strict system prompt constraints, extracts answers from structural XML tags (`<answer>...</answer>`), and executes symbolic error verification.
3. **`src/evaluator.py`**: The orchestrator layer. Loops through the evaluation dataset, feeds instances to the pipeline, manages API rate-limits, tracks performance metrics, and handles real-time state persistence to disk.

---

## 🔄 The Symbolic Feedback Loop Flow

```text
[Dataset Entry] ──> [LLM Round 1 Attempt] ──> [Structural Extraction]
                                                       │
                                                       ▼
[Success / Log] <── [✅ Match] <── [SymPy Structural Equivalency Check]
                                                       │
                                                       ▼
[Final Fail]    <── [❌ Fail]  <── [LLM Round 2 Attempt] <── [Mathematical Reprompt]
```

Instead of naive string-matching, the verification layer utilizes structural mathematical properties. A response is only marked correct if: 
`Simplify(LLM_Output - Ground_Truth) == 0`

If this condition fails or a parsing error occurs (such as the LLM utilizing LaTeX instead of raw computational strings), a structured critique is compiled and injected back into the conversation context for a second-stage correction attempt.

---

## 📊 Key Performance Metrics Measured

The evaluation harness evaluates system accuracy using three primary statistical metrics:
* **Baseline (0-Shot) Accuracy**: The percentage of problems the LLM solves perfectly on its first attempt.
* **Correction Rate**: The percentage of initially failed problems that the system successfully recovers via the feedback loop.
* **Final System Accuracy**: The comprehensive performance of the entire system (Baseline Successes + Recovered Successes).

---

## 🚀 Getting Started (Setup & Execution)

### Prerequisites
This project utilizes `uv`, a fast, modern Python package installer and resolver. Ensure you have it installed:
```bash
curl -fsSL [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

### 1. Installation
Clone the repository and install dependencies using the virtual environment manager:
```bash
# Clone the repository
git clone git@github.com:Uday232523/calculus-verifier.git
cd calculus-verifier

# Sync dependencies 
uv sync
```

### 2. Environment Configuration
The system isolates sensitive access tokens using environment variables. Create a `.env` file in the project root:

```env
# For Hugging Face Serverless Architecture
HF_TOKEN="your_huggingface_access_token_here"

# For Google Gemini Architecture (Fallback)
GEMINI_API_KEY="your_gemini_api_key_here"
```
*(Note: `.env` is explicitly sandboxed in `.gitignore` to protect credentials from remote exposure).*

### 3. Running the Framework

**Step 1: Generate the Problem Dataset**
```bash
uv run src/generator.py
```
This builds a balanced dataset of complex functions (including quotient rule and chain rule problems) inside `data/problems.csv`.

**Step 2: Run the Evaluation Loop**
```bash
uv run src/evaluator.py
```
The evaluator will process the dataset sequentially, display live verification logs, run the stateful correction loops when an error is trapped, and save the full granular trial details directly to `data/results.csv`.

---

## 🛠️ Tech Stack & Core Libraries

* **Language Layer**: Python 3.12+
* **Symbolic Math Engine**: `SymPy` (Symbolic computation, expression parsing, and algebraic simplification)
* **Data Automation**: `Pandas` (Dataset compilation, processing, and CSV synchronization)
* **Execution & Resolution**: `uv` + `python-dotenv`
* **Inference Routing Layer**: `OpenAI SDK` / `Google GenAI SDK` pointing to Hugging Face Inference Router
