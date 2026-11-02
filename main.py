"""
main.py
=======
Entry point for the AI-Optimized Fmax Search project.

Usage:
    python main.py --help
    python main.py validate
    python main.py train
    python main.py evaluate
    python main.py predict --ro 0.1 -0.2 0.3 -0.1 0.4 -0.3 0.2 -0.1 0.3 -0.2 \
                           --iddq 0.5 --vmin -0.3 --die_x 0.4 --die_y 0.1 --radius_sq 0.17
    python main.py all      # runs validate + train + evaluate
"""

import argparse
import sys
from pathlib import Path

# Project root — add to path so 'src' imports work
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Locate the dataset
def _find_csv() -> Path:
    candidates = [
        PROJECT_ROOT / "synthetic_fmax_dataset.csv",
        PROJECT_ROOT / "data" / "synthetic_fmax_dataset.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find 'synthetic_fmax_dataset.csv'.\n"
        "Expected locations:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )


MODEL_DIR  = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_validate(args) -> int:
    from src.data_loader import load_dataset
    from src.data_validation import run_all_validations

    csv_path = _find_csv()
    print(f"Validating dataset: {csv_path}")
    df = load_dataset(csv_path)
    ok = run_all_validations(df, verbose=True)
    return 0 if ok else 1


def cmd_train(args) -> int:
    from src.train_model import run_training

    csv_path = _find_csv()
    print(f"Training model on dataset: {csv_path}")
    run_training(csv_path, model_dir=MODEL_DIR)
    print("\n[OK] Training complete. Model saved to: models/")
    return 0


def cmd_evaluate(args) -> int:
    from src.evaluate_model import run_evaluation

    csv_path = _find_csv()
    if not (MODEL_DIR / "gbr_model.joblib").exists():
        print("[ERROR] Model not found. Run 'python main.py train' first.")
        return 1

    print(f"Evaluating model on dataset: {csv_path}")
    run_evaluation(csv_path, model_dir=MODEL_DIR, output_dir=OUTPUT_DIR)
    print("\n[OK] Evaluation complete. Results saved to: outputs/")
    return 0


def cmd_predict(args) -> int:
    if not (MODEL_DIR / "gbr_model.joblib").exists():
        print("[ERROR] Model not found. Run 'python main.py train' first.")
        return 1

    # Build a namespace compatible with predict.main()
    from src.predict import build_feature_df, predict_fmax, print_prediction
    import json

    if args.csv_row:
        import pandas as pd
        from src.data_loader import FEATURE_COLS
        feature_df = pd.read_csv(args.csv_row)[FEATURE_COLS]
    elif args.ro is not None:
        required_attrs = {"iddq", "vmin", "die_x", "die_y", "radius_sq"}
        missing = [a for a in required_attrs if getattr(args, a, None) is None]
        if missing:
            print(f"❌ Missing arguments: {', '.join('--'+m for m in missing)}")
            return 1
        feature_df = build_feature_df(
            ro_values  = args.ro,
            iddq_norm  = args.iddq,
            vmin_norm  = args.vmin,
            die_x      = args.die_x,
            die_y      = args.die_y,
            radius_sq  = args.radius_sq,
        )
    else:
        print("❌ For 'predict', supply --ro (10 values) + --iddq --vmin --die_x --die_y --radius_sq, OR --csv_row.")
        return 1

    result = predict_fmax(feature_df, model_dir=MODEL_DIR)
    if getattr(args, "json_out", False):
        print(json.dumps(result, indent=2))
    else:
        print_prediction(result)
    return 0


def cmd_all(args) -> int:
    for fn in [cmd_validate, cmd_train, cmd_evaluate]:
        rc = fn(args)
        if rc != 0:
            return rc
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="AI-Optimized Fmax Search for Test Time Reduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  validate   Validate the dataset schema, splits, and leakage checks.
  train      Train the GradientBoostingRegressor model and calibrate.
  evaluate   Evaluate the trained model on the test set and generate plots.
  predict    Predict Fmax for a new chip from pre-test measurements.
  all        Run validate → train → evaluate in sequence.

Examples:
  python main.py all
  python main.py train
  python main.py evaluate
  python main.py predict --ro 0.1 -0.2 0.3 -0.1 0.4 -0.3 0.2 -0.1 0.3 -0.2 \\
                         --iddq 0.5 --vmin -0.3 --die_x 0.4 --die_y 0.1 --radius_sq 0.17
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # validate
    subparsers.add_parser("validate", help="Validate dataset schema and integrity.")

    # train
    subparsers.add_parser("train", help="Train and calibrate the GBR model.")

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate model and generate all outputs.")

    # predict
    pred_p = subparsers.add_parser("predict", help="Predict Fmax for a new chip.")
    pred_p.add_argument("--ro", nargs=10, type=float, metavar="V",
                        help="10 ring-oscillator values (ro_01 … ro_10)")
    pred_p.add_argument("--iddq",      type=float, help="Normalised IDDQ")
    pred_p.add_argument("--vmin",      type=float, help="Normalised Vmin")
    pred_p.add_argument("--die_x",     type=float, help="Die X on wafer")
    pred_p.add_argument("--die_y",     type=float, help="Die Y on wafer")
    pred_p.add_argument("--radius_sq", type=float, help="Squared wafer radius")
    pred_p.add_argument("--csv_row",   type=str,   help="CSV file with one measurement row")
    pred_p.add_argument("--json",      dest="json_out", action="store_true",
                        help="Output as JSON")

    # all
    subparsers.add_parser("all", help="Run validate + train + evaluate.")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    command_map = {
        "validate": cmd_validate,
        "train":    cmd_train,
        "evaluate": cmd_evaluate,
        "predict":  cmd_predict,
        "all":      cmd_all,
    }

    fn = command_map.get(args.command)
    if fn is None:
        parser.print_help()
        return 1

    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
