"""Fresh-clone demo bootstrap.

Runs the full synthetic pipeline end-to-end so ANY clone can get a working
score without downloading CWRU, building Docker images, or having pre-trained
models checked in:

    generate data -> features -> train anomaly + fault -> promote (production)
    -> print a ready-to-run curl command.

This is the "interviewer clones the repo" path: no CWRU download, no Docker,
no pre-trained models. Everything is deterministic (fixed seeds), so a fresh
clone can reproduce the artifacts exactly. The promotion gate evaluates the
``val`` split, which ``aether_pdm.data.synthetic.generate_dataset`` always
emits (normal + faulty rows), so promote works without any CWRU data.

Usage:
    uv run python scripts/bootstrap_demo.py
    # then start the API in another terminal:
    #   uv run uvicorn aether_pdm.serve.app:app --port 8000
    # and run the printed curl.
"""

import argparse
import os
import sys
from pathlib import Path

from aether_pdm.data.synthetic import generate_dataset, synthetic_waveform
from aether_pdm.models.anomaly import train_anomaly
from aether_pdm.models.fault import train_fault_classifier
from aether_pdm.ops.promote import promote_anomaly, promote_fault
from aether_pdm.signal.pipeline import FEATURE_VERSION, process_dataset

DEFAULT_WORKDIR = Path("data/demo_bootstrap")

# Seed used for the whole deterministic demo run.
_DEMO_SEED = 42


def _print_curl() -> None:
    """Print the ready-to-run API + curl instructions."""
    # A real snippet from a deterministic fault waveform so the printed curl
    # contains actual signal values (truncated to keep it printable).
    snippet = synthetic_waveform(
        length=2048,
        fault_type="inner_race",
        fault_diameter=0.021,
        seed=_DEMO_SEED,
    ).tolist()
    snippet_str = ", ".join(f"{v:.6f}" for v in snippet[:12])

    print()
    print("=" * 72)
    print("Bootstrap complete.")
    print("=" * 72)
    print()
    print("Start the API in another terminal:")
    print("  uv run uvicorn aether_pdm.serve.app:app --port 8000")
    print()
    print("Then run a scoring request (2048 samples @ 12 kHz; paste/repeat the")
    print("snippet below to fill the waveform list, or generate one with")
    print("  uv run python -c \"import json; from aether_pdm.data.synthetic import"
    " synthetic_waveform; print(json.dumps({'waveform': synthetic_waveform(2048,"
    " fault_type='inner_race', fault_diameter=0.021, seed=1).tolist(),"
    " 'sampling_rate': 12000}))\")")
    print("  curl -s -X POST http://localhost:8000/v1/assets/synth-demo/score \\")
    print("    -H 'Content-Type: application/json' \\")
    print(f"    -d '{{\"waveform\": [{snippet_str}, ...], \"sampling_rate\": 12000}}'")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORKDIR,
        help="Working directory for all artifacts (default: data/demo_bootstrap)",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: sqlite:///mlflow.db inside --workdir)",
    )
    parser.add_argument("--n-normal", type=int, default=10)
    parser.add_argument("--n-faulty", type=int, default=12)
    parser.add_argument("--n-estimators", type=int, default=50)
    args = parser.parse_args(argv)

    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    mlflow_uri = args.mlflow_uri or "sqlite:///mlflow.db"

    # Keep every artifact (parquet, sqlite db, mlruns/) inside the workdir so
    # the demo never pollutes the repo tree. Restore cwd on the way out.
    prev_cwd = Path.cwd()
    os.chdir(workdir)
    try:
        print(f"Workdir: {workdir}")
        print(f"MLflow URI: {mlflow_uri}")

        # 1. Generate synthetic data (deterministic, fixed seed). The generator
        #    emits train/val/test splits with the val split guaranteed to hold
        #    both normal AND faulty samples.
        data_path = generate_dataset(
            workdir / "synth",
            n_normal=args.n_normal,
            n_faulty=args.n_faulty,
            seed=_DEMO_SEED,
        )

        # 2. Signal pipeline -> features parquet (carries the split column).
        feat_dir = workdir / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        process_dataset(data_path, output_dir=feat_dir, window_size=2048, overlap=0.5)
        features_path = feat_dir / f"features_{FEATURE_VERSION}.parquet"

        # 3. Train anomaly detector (IsolationForest on healthy windows).
        #    strict_boundary=True pins the decision threshold at the worst
        #    healthy window: training on PURE healthy data with contamination>0
        #    would otherwise force ~contamination of healthy rows below the
        #    boundary, guaranteeing a false-alarm floor (~FAR>=0.1) on the val
        #    healthy set and failing the promotion gate. With the strict
        #    boundary, val healthy windows score >= 0 (FAR=0) while synthetic
        #    fault windows (far outside the healthy hull) score negative.
        print("\n=== Training anomaly detector ===")
        train_anomaly(
            features_path,
            contamination=0.1,
            n_estimators=args.n_estimators,
            random_state=_DEMO_SEED,
            mlflow_uri=mlflow_uri,
            strict_boundary=True,
        )

        # 4. Train fault classifier (RandomForest on all labeled windows).
        print("\n=== Training fault classifier ===")
        train_fault_classifier(
            features_path,
            n_estimators=args.n_estimators,
            max_depth=8,
            random_state=_DEMO_SEED,
            mlflow_uri=mlflow_uri,
        )

        # 5. Promote to production through the GatedOps gate. The gate
        #    evaluates the val split (DEFAULT_SPLIT="val"); on deterministic
        #    synthetic data the classes separate cleanly and the gates pass.
        print("\n=== Promoting anomaly model ===")
        anomaly_result = promote_anomaly(features_path, mlflow_uri=mlflow_uri)
        print(f"  decision={anomaly_result['decision']} ({anomaly_result['reason']})")

        print("\n=== Promoting fault model ===")
        fault_result = promote_fault(features_path, mlflow_uri=mlflow_uri)
        print(f"  decision={fault_result['decision']} ({fault_result['reason']})")

        if (
            anomaly_result["decision"] != "promoted"
            or fault_result["decision"] != "promoted"
        ):
            print("WARNING: one or more promotion gates did not pass.", file=sys.stderr)
            print(
                "The API may still serve the latest non-production model.",
                file=sys.stderr,
            )
            return 1

        _print_curl()
    finally:
        os.chdir(prev_cwd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
