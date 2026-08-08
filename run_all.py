import subprocess
import sys

STEPS = [
    ("Problem 1: distance matrices", ["python3", "-m", "src.distances"]),
    ("Problem 2: polynomial fitting", ["python3", "-m", "src.poly_fitting"]),
    ("Problem 3: bias-variance decomposition", ["python3", "-m", "src.bias_variance"]),
    ("Problem 4: California Housing EDA", ["python3", "-m", "src.eda"]),
    ("Problem 6 (optional): ridge regression", ["python3", "-m", "src.ridge"]),
]

def main():
    for name, cmd in STEPS:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {name}", file=sys.stderr)
            sys.exit(result.returncode)
    print("\nAll steps completed. See figures/ and results/ for outputs.")

if __name__ == "__main__":
    main()
