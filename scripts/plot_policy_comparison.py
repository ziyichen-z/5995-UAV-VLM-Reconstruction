"""
plot_policy_comparison.py

Aggregate VLM-vs-heuristic selection-policy comparison outputs and plot
mean coverage by seed.

Example:
  python3 scripts/plot_policy_comparison.py \
    --input-root outputs/policy_benchmark_20260504_120000 \
    --output-dir outputs/policy_benchmark_20260504_120000/summary
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True,
                        help="Directory containing run subdirectories.")
    parser.add_argument("--output-dir", default=None,
                        help="Where to write CSVs and PNG. Defaults to input-root/summary.")
    parser.add_argument("--require-equal-visits", action="store_true",
                        help="Drop runs where VLM and heuristic used different visit counts.")
    parser.add_argument("--basis", choices=["equal-visits", "budget-final"],
                        default="equal-visits",
                        help="Plot equal actual visits or final coverage after each policy budget.")
    return parser.parse_args()


def iter_comparison_files(input_root: Path):
    for path in sorted(input_root.rglob("selection_policy_comparison.json")):
        if "/comparison/vlm/" in str(path) or "/comparison/heuristic/" in str(path):
            continue
        yield path


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_run_dir(comparison_path: Path) -> Path:
    # .../<run_dir>/comparison/selection_policy_comparison.json
    return comparison_path.parents[1]


def room_size(run_dir: Path):
    manifest_path = run_dir / "manifests" / "room_manifest.json"
    if not manifest_path.exists():
        return None, None, None, ""
    manifest = load_json(manifest_path)
    room = manifest.get("room", {})
    width = room.get("width")
    depth = room.get("depth")
    height = room.get("height")
    if width is None or depth is None or height is None:
        bounds = room.get("bounds", {})
        xb = bounds.get("x", [None, None])
        yb = bounds.get("y", [None, None])
        zb = bounds.get("z", [None, None])
        width = xb[1] - xb[0] if None not in xb else None
        depth = yb[1] - yb[0] if None not in yb else None
        height = zb[1] - zb[0] if None not in zb else None
    label = ""
    if width is not None and depth is not None and height is not None:
        label = f"{width:.2f} x {depth:.2f} x {height:.2f}"
    return width, depth, height, label


def seed_for_run(run_dir: Path):
    cfg_path = run_dir / "pipeline_config.json"
    if cfg_path.exists():
        cfg = load_json(cfg_path)
        return cfg.get("scene", {}).get("seed")
    for part in run_dir.name.split("_"):
        if part.startswith("seed"):
            try:
                return int(part.replace("seed", ""))
            except ValueError:
                pass
    return None


def records_from_outputs(input_root: Path):
    rows = []
    for comparison_path in iter_comparison_files(input_root):
        run_dir = find_run_dir(comparison_path)
        comparison = load_json(comparison_path)
        policies = comparison.get("policies", {})
        if "vlm" not in policies or "heuristic" not in policies:
            continue

        seed = seed_for_run(run_dir)
        width, depth, height, room_label = room_size(run_dir)
        vlm = policies["vlm"]
        heuristic = policies["heuristic"]
        initial = comparison.get("initial_coverage")

        vlm_steps = vlm.get("step_logs", [])
        heuristic_steps = heuristic.get("step_logs", [])
        vlm_visits = vlm.get("visits_used") or 0
        heuristic_visits = heuristic.get("visits_used") or 0
        equal_visits = min(vlm_visits, heuristic_visits)

        def coverage_at(policy_summary: dict, steps: list, n_visits: int):
            if n_visits <= 0:
                return initial
            if len(steps) >= n_visits:
                return steps[n_visits - 1].get("coverage_after")
            return policy_summary.get("final_coverage")

        vlm_equal = coverage_at(vlm, vlm_steps, equal_visits)
        heuristic_equal = coverage_at(heuristic, heuristic_steps, equal_visits)

        rows.append({
            "seed": seed,
            "run_dir": str(run_dir),
            "room_width_m": width,
            "room_depth_m": depth,
            "room_height_m": height,
            "room_size_m": room_label,
            "initial_coverage": initial,
            "vlm_final_coverage": vlm.get("final_coverage"),
            "heuristic_final_coverage": heuristic.get("final_coverage"),
            "delta_vlm_minus_heuristic": (
                vlm.get("final_coverage", 0) - heuristic.get("final_coverage", 0)
            ),
            "equalized_visits": equal_visits,
            "vlm_equal_visit_coverage": vlm_equal,
            "heuristic_equal_visit_coverage": heuristic_equal,
            "equal_visit_delta_vlm_minus_heuristic": (
                (vlm_equal or 0) - (heuristic_equal or 0)
            ),
            "vlm_gain": vlm.get("coverage_gain"),
            "heuristic_gain": heuristic.get("coverage_gain"),
            "vlm_visits_used": vlm_visits,
            "heuristic_visits_used": heuristic_visits,
            "vlm_frames_used": vlm.get("frames_used"),
            "heuristic_frames_used": heuristic.get("frames_used"),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, basis: str) -> pd.DataFrame:
    if basis == "equal-visits":
        vlm_col = "vlm_equal_visit_coverage"
        heuristic_col = "heuristic_equal_visit_coverage"
        delta_col = "equal_visit_delta_vlm_minus_heuristic"
    else:
        vlm_col = "vlm_final_coverage"
        heuristic_col = "heuristic_final_coverage"
        delta_col = "delta_vlm_minus_heuristic"

    grouped = df.groupby("seed", dropna=False)
    summary = grouped.agg(
        runs=("run_dir", "count"),
        room_size_m=("room_size_m", "first"),
        initial_mean=("initial_coverage", "mean"),
        initial_std=("initial_coverage", "std"),
        vlm_mean=(vlm_col, "mean"),
        vlm_std=(vlm_col, "std"),
        heuristic_mean=(heuristic_col, "mean"),
        heuristic_std=(heuristic_col, "std"),
        delta_mean=(delta_col, "mean"),
        delta_std=(delta_col, "std"),
        equalized_visits_mean=("equalized_visits", "mean"),
        vlm_gain_mean=("vlm_gain", "mean"),
        heuristic_gain_mean=("heuristic_gain", "mean"),
        vlm_visits_mean=("vlm_visits_used", "mean"),
        heuristic_visits_mean=("heuristic_visits_used", "mean"),
        vlm_frames_mean=("vlm_frames_used", "mean"),
        heuristic_frames_mean=("heuristic_frames_used", "mean"),
    ).reset_index()
    return summary.sort_values("seed")


def pct(series):
    return series.astype(float) * 100.0


def plot_summary(summary: pd.DataFrame, output_path: Path, basis: str):
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = range(len(summary))
    width = 0.34

    vlm_mean = pct(summary["vlm_mean"])
    heuristic_mean = pct(summary["heuristic_mean"])
    vlm_std = pct(summary["vlm_std"].fillna(0))
    heuristic_std = pct(summary["heuristic_std"].fillna(0))
    initial = pct(summary["initial_mean"])

    ax.bar([i - width / 2 for i in x], vlm_mean, width,
           yerr=vlm_std, capsize=4, label="VLM", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], heuristic_mean, width,
           yerr=heuristic_std, capsize=4, label="Heuristic", color="#F58518")
    ax.plot(list(x), initial, color="#555555", marker="o",
            linewidth=1.7, label="Initial coverage")

    for i, row in summary.iterrows():
        y = max(row["vlm_mean"], row["heuristic_mean"]) * 100.0 + 2.5
        delta = row["delta_mean"] * 100.0
        ax.text(i, y, f"Δ {delta:+.1f} pts", ha="center", va="bottom",
                fontsize=9, color="#222222")

    labels = [str(int(s)) if pd.notna(s) else "unknown" for s in summary["seed"]]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coverage (%)")
    ax.set_xlabel("Seed")
    title_suffix = "Equal Actual Visits" if basis == "equal-visits" else "Final Budget Result"
    ax.set_title(f"VLM vs Heuristic Supplementary Capture ({title_suffix})")
    ax.set_ylim(0, max(100, float(max(vlm_mean.max(), heuristic_mean.max(), initial.max())) + 10))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")

    if basis == "equal-visits":
        note = "Fairness basis: coverage compared after the same actual number of visits per run."
    else:
        note = "Fairness basis: final coverage after each policy's configured visit budget."
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = records_from_outputs(input_root)
    if df.empty:
        raise SystemExit(f"No VLM+heuristic comparison outputs found under {input_root}")

    if args.require_equal_visits:
        before = len(df)
        df = df[df["vlm_visits_used"] == df["heuristic_visits_used"]]
        print(f"[plot_policy_comparison] Equal-visit filter: {len(df)}/{before} runs kept.")
        if df.empty:
            raise SystemExit("No runs remain after equal-visit filtering.")

    summary = summarize(df, args.basis)
    per_run_csv = output_dir / "policy_comparison_runs.csv"
    summary_csv = output_dir / "policy_comparison_summary.csv"
    plot_png = output_dir / "policy_comparison.png"

    df.to_csv(per_run_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    plot_summary(summary, plot_png, args.basis)

    print(f"[plot_policy_comparison] Wrote {per_run_csv}")
    print(f"[plot_policy_comparison] Wrote {summary_csv}")
    print(f"[plot_policy_comparison] Wrote {plot_png}")


if __name__ == "__main__":
    main()
