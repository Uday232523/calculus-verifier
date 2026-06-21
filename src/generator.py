import os
import random
import sympy as sp
import pandas as pd


def generate_dataset(target_count=100):
    # Define our variable
    x = sp.Symbol("x")

    # A pool of basic functions to mix and match
    base_funcs = [x, x**2, x**3, sp.sin(x), sp.cos(x), sp.exp(x)]

    generated_data = []
    seen_expressions = set()

    print(f"⚙️ Generating {target_count} unique calculus problems...")

    while len(generated_data) < target_count:
        # Pick two random base functions
        f = random.choice(base_funcs)
        g = random.choice(base_funcs)

        # Pick a random differentiation rule to apply
        rule = random.choice(["product", "quotient", "chain", "product_chain"])

        try:
            if rule == "product":
                expr = f * g
            elif rule == "quotient":
                # Add a constant to the denominator to avoid division by zero errors
                expr = f / (g + random.randint(1, 3))
            elif rule == "chain":
                # Substitute x in function 'f' with function 'g' (e.g., f(g(x)))
                expr = f.subs(x, g)
            elif rule == "product_chain":
                # e.g., x^3 * exp(sin(x))
                h = random.choice(base_funcs)
                expr = f * g.subs(x, h)

            # Skip if the expression is trivial (just 'x' or a constant number)
            if expr == x or expr.is_constant():
                continue

            expr_str = str(expr)

            # Skip if we've already generated this exact problem
            if expr_str in seen_expressions:
                continue

            # THE MAGIC: Let SymPy calculate the perfect ground-truth derivative!
            derivative = sp.diff(expr, x)

            if derivative == 0:
                continue

            seen_expressions.add(expr_str)
            generated_data.append(
                {
                    "problem_id": f"P{len(generated_data) + 1:03d}",
                    "rule": rule,
                    "expression": expr_str,
                    "expression_latex": sp.latex(expr),
                    "ground_truth": str(derivative),
                    "ground_truth_latex": sp.latex(derivative),
                }
            )

        except Exception:
            # If SymPy fails to construct something valid, just skip it
            continue

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)

    # Save to a CSV using Pandas
    df = pd.DataFrame(generated_data)
    df.to_csv("data/problems.csv", index=False)

    print(f"✅ Successfully saved {len(df)} problems to 'data/problems.csv'")
    print(df.head())  # Show a preview


if __name__ == "__main__":
    generate_dataset(100)
