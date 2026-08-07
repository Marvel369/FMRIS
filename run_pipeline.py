"""
run_pipeline.py
----------------
Runs your whole :-> data_loader -> preprocessing -> feature_engineering -> train_model

Usage:
    python run_pipeline.py
"""

import subprocess
import sys

STEPS = [
    "src/data_loader.py",
    "src/preprocessing.py",
    "src/feature_engineering.py",
    "src/train_model.py",
]

for step in STEPS:
    print(f"\n{'='*50}\nRunning {step}\n{'='*50}")
    result = subprocess.run([sys.executable, step])
    if result.returncode != 0:
        print(f"\n{step} failed — stopping pipeline.")
        sys.exit(1)

print("\n---\nPipeline complete!!!\n Run `python src/model_selector.py` next to pick the best model,")
print("then `python src/predictor.py` to get tomorrow's risk probability.\n---\n")