"""
plot_heuristic_summary.py

Aggregate heuristic-only supplementary capture results and plot mean coverage.
Accepts outputs produced either by --selection-policy heuristic or by
--selection-policy both.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_comparison_files(input_root: Path):
    for path in sorted(input_root.rglob("selection_policy_comparison.json")):
        if "/comparison/heuristic/" in str(path) or "/comparison/vlm/" in str(path):
            continue
        yield path


def run_dir_for_comparison(path: Path) -> Path:
    return path.parents[1]


def seed_for_run(run_dir: Path):
    cfg_path = run_dir / "pipeline_config.json"
    if cfg_path.exists():
        return load_json(cfg_path).get("scene", {}).get("seed")
    for part in run_dir.name.split("_"):
        if part.startswith("seed"):
            try:
                return int(part.replace("seed", ""))
            except ValueError:
                pass
    return None


def room_size(run_dir: Path):
    manifest_path = run_dir / "manifests" / "room_manifest.json"
    if not manifest_path.exists():
        return ""
    room = load_json(manifest_path).get("room", {})
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
    if width is None or depth is None or height is None:
        return ""
    return f"{width:.2f} x {depth:.2f} x {height:.2f}"


def collect(input_root: Path, seeds: Optional[set]):
    rows = []
    for comparison_path in iter_comparison_files(input_root):
        run_dir = run_dir_for_comparison(comparison_path)
        comparison = load_json(comparison_path)
        heuristic = comparison.get("policies", {}).get("heuristic")
        if not heuristic:
            continue
        seed = seed_for_run(run_dir)
        if seeds is not None and seed not in seeds:
            continue
        initial = comparison.get("initial_coverage")
        final = heuristic.get("final_coverage")
        rows.append({
            "seed": seed,
            "run_dir": str(run_dir),
            "room_size_m": room_size(run_dir),
            "initial_coverage": initial,
            "heuristic_final_coverage": final,
            "heuristic_gain": final - initial if final is not None and initial is not None else None,
            "heuristic_visits_used": heuristic.get("visits_used"),
            "heuristic_frames_used": heuristic.get("frames_used"),
        })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    by_seed = df.groupby("seed", dropna=False).agg(
        runs=("run_dir", "count"),
        room_size_m=("room_size_m", "first"),
        initial_mean=("initial_coverage", "mean"),
        initial_std=("initial_coverage", "std"),
        heuristic_mean=("heuristic_final_coverage", "mean"),
        heuristic_std=("heuristic_final_coverage", "std"),
        heuristic_gain_mean=("heuristic_gain", "mean"),
        heuristic_gain_std=("heuristic_gain", "std"),
        heuristic_visits_mean=("heuristic_visits_used", "mean"),
        heuristic_frames_mean=("heuristic_frames_used", "mean"),
    ).reset_index().sort_values("seed")

    overall = pd.DataFrame([{
        "seed": "ALL",
        "runs": len(df),
        "room_size_m": "",
        "initial_mean": df["initial_coverage"].mean(),
        "initial_std": df["initial_coverage"].std(),
        "heuristic_mean": df["heuristic_final_coverage"].mean(),
        "heuristic_std": df["heuristic_final_coverage"].std(),
        "heuristic_gain_mean": df["heuristic_gain"].mean(),
        "heuristic_gain_std": df["heuristic_gain"].std(),
        "heuristic_visits_mean": df["heuristic_visits_used"].mean(),
        "heuristic_frames_mean": df["heuristic_frames_used"].mean(),
    }])
    return by_seed, overall


def plot(by_seed: pd.DataFrame, overall: pd.DataFrame, output_path: Path):
    labels = [str(int(s)) if pd.notna(s) else "unknown" for s in by_seed["seed"]]
    labels.append("ALL")

    coverage = list(by_seed["heuristic_mean"] * 100.0)
    coverage.append(float(overall["heuristic_mean"].iloc[0] * 100.0))
    err = list(by_seed["heuristic_std"].fillna(0) * 100.0)
    err.append(float((overall["heuristic_std"].fillna(0).iloc[0]) * 100.0))

    initial = list(by_seed["initial_mean"] * 100.0)
    initial.append(float(overall["initial_mean"].iloc[0] * 100.0))

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(labels))
    colors = ["#F58518"] * (len(labels) - 1) + ["#54A24B"]
    ax.bar(x, coverage, yerr=err, capsize=4, color=colors, label="Heuristic final")
    ax.plot(list(x), initial, color="#555555", marker="o", label="Initial")

    for idx, value in enumerate(coverage):
        ax.text(idx, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coverage (%)")
    ax.set_xlabel("Seed")
    ax.set_title("Heuristic Supplementary Capture: Seed Means and Overall Average")
    ax.set_ylim(0, max(100, max(coverage + initial) + 10))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_readable_summary(df: pd.DataFrame, by_seed: pd.DataFrame,
                           overall: pd.DataFrame, output_dir: Path):
    rounded_rows = []
    for _, row in by_seed.iterrows():
        rounded_rows.append({
            "seed": int(row["seed"]) if pd.notna(row["seed"]) else "unknown",
            "room_size_m": row["room_size_m"],
            "runs": int(row["runs"]),
            "initial_coverage_pct": round(row["initial_mean"] * 100.0, 2),
            "initial_std_pct_pts": round(
                0.0 if pd.isna(row["initial_std"]) else row["initial_std"] * 100.0, 2
            ),
            "heuristic_coverage_pct": round(row["heuristic_mean"] * 100.0, 2),
            "heuristic_std_pct_pts": round(
                0.0 if pd.isna(row["heuristic_std"]) else row["heuristic_std"] * 100.0, 2
            ),
            "gain_percentage_points": round(row["heuristic_gain_mean"] * 100.0, 2),
            "gain_std_pct_pts": round(
                0.0 if pd.isna(row["heuristic_gain_std"]) else row["heuristic_gain_std"] * 100.0, 2
            ),
            "visits_mean": round(row["heuristic_visits_mean"], 2),
            "frames_mean": round(row["heuristic_frames_mean"], 2),
        })

    overall_row = overall.iloc[0]
    rounded_rows.append({
        "seed": "ALL",
        "room_size_m": "",
        "runs": int(overall_row["runs"]),
        "initial_coverage_pct": round(overall_row["initial_mean"] * 100.0, 2),
        "initial_std_pct_pts": round(overall_row["initial_std"] * 100.0, 2),
        "heuristic_coverage_pct": round(overall_row["heuristic_mean"] * 100.0, 2),
        "heuristic_std_pct_pts": round(overall_row["heuristic_std"] * 100.0, 2),
        "gain_percentage_points": round(overall_row["heuristic_gain_mean"] * 100.0, 2),
        "gain_std_pct_pts": round(overall_row["heuristic_gain_std"] * 100.0, 2),
        "visits_mean": round(overall_row["heuristic_visits_mean"], 2),
        "frames_mean": round(overall_row["heuristic_frames_mean"], 2),
    })

    rounded = pd.DataFrame(rounded_rows)
    rounded.to_csv(output_dir / "heuristic_summary_rounded.csv", index=False)

    n_seeds = by_seed["seed"].nunique()
    n_runs = int(overall_row["runs"])
    initial_pct = overall_row["initial_mean"] * 100.0
    heuristic_pct = overall_row["heuristic_mean"] * 100.0
    gain_pts = overall_row["heuristic_gain_mean"] * 100.0
    visits_mean = overall_row["heuristic_visits_mean"]
    frames_mean = overall_row["heuristic_frames_mean"]

    table_lines = [
        "| Seed | Runs | Room Size (m) | Initial Coverage | Heuristic Coverage | Gain | Visits | Frames |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in rounded.iloc[:-1].iterrows():
        table_lines.append(
            f"| {row['seed']} | {row['runs']} | {row['room_size_m']} | "
            f"{row['initial_coverage_pct']:.2f}% +/- {row['initial_std_pct_pts']:.2f} | "
            f"{row['heuristic_coverage_pct']:.2f}% +/- {row['heuristic_std_pct_pts']:.2f} | "
            f"{row['gain_percentage_points']:+.2f} pts | "
            f"{row['visits_mean']:.2f} | {row['frames_mean']:.2f} |"
        )
    table_lines.append(
        f"| ALL | {n_runs} |  | "
        f"{rounded.iloc[-1]['initial_coverage_pct']:.2f}% +/- "
        f"{rounded.iloc[-1]['initial_std_pct_pts']:.2f} | "
        f"{rounded.iloc[-1]['heuristic_coverage_pct']:.2f}% +/- "
        f"{rounded.iloc[-1]['heuristic_std_pct_pts']:.2f} | "
        f"{rounded.iloc[-1]['gain_percentage_points']:+.2f} pts | "
        f"{rounded.iloc[-1]['visits_mean']:.2f} | "
        f"{rounded.iloc[-1]['frames_mean']:.2f} |"
    )

    markdown = f"""# Heuristic Supplementary Capture Summary

This heuristic-only benchmark used {n_seeds} seeds with {n_runs} total runs. The heuristic was configured for up to 3 supplementary visits per run. In the current forced-budget setting, strict heuristic candidates are used first; if those candidates are exhausted, deterministic object or room-geometry fallback candidates are used so each run can spend the same visit budget.

## Formulas

Coverage gain for run `i`:

```text
Delta C_i = C_heuristic_final_i - C_initial_i
```

Mean coverage across `N` runs:

```text
mean(C) = (1 / N) * sum(C_i)
```

Sample standard deviation across runs:

```text
std(C) = sqrt(sum((C_i - mean(C))^2) / (N - 1))
```

Mean gain and percent gain relative to the initial coverage:

```text
mean(Delta C) = (1 / N) * sum(Delta C_i)
relative_gain_pct = 100 * mean(Delta C) / mean(C_initial)
```

Mean visits and frames:

```text
mean_visits = (1 / N) * sum(visits_i)
mean_frames = (1 / N) * sum(frames_i)
```

For a same-budget comparison with VLM, both policies should be evaluated using the same seed, same initial capture, same maximum visit count, and same frames per visit:

```text
Delta_policy = C_policy_final - C_initial
Delta(VLM - Heuristic) = C_VLM_final - C_heuristic_final
```

## Results

{chr(10).join(table_lines)}

## Poster Summary

We implemented a non-VLM heuristic supplementary capture policy as a deterministic baseline for active scene exploration. The heuristic ranks under-covered spatial regions using coverage density, view-angle diversity, object overlap, and room-position cues such as corners or edges, then selects high-priority candidates for additional orbit captures. Unlike the VLM policy, this method does not inspect image content semantically; it relies only on pose logs, room geometry, and coverage statistics, making it inexpensive, reproducible, and useful as a control condition. To make the comparison budget-matched, the heuristic was configured for three supplementary visits per run; strict heuristic candidates were prioritized, with deterministic fallback candidates used only when needed to spend the same visit budget. Across these runs, the heuristic improved average coverage from {initial_pct:.2f}% to {heuristic_pct:.2f}%, corresponding to a mean gain of {gain_pts:+.2f} percentage points. These results suggest that the heuristic can recover some missed coverage, while also providing a controlled baseline for testing whether VLM-guided selection adds value beyond geometric coverage optimization.

## Overall

Across all runs, mean initial coverage was **{initial_pct:.2f}%** and mean heuristic final coverage was **{heuristic_pct:.2f}%**. The mean heuristic gain was **{gain_pts:+.2f} percentage points**. The heuristic used an average of **{visits_mean:.2f} visits** and **{frames_mean:.2f} supplementary frames** per run.
"""
    with open(output_dir / "heuristic_summary.md", "w", encoding="utf-8") as f:
        f.write(markdown)


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_root / "heuristic_summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = set(args.seeds) if args.seeds else None

    df = collect(input_root, seeds)
    if df.empty:
        raise SystemExit(f"No heuristic comparison outputs found under {input_root}")

    by_seed, overall = summarize(df)
    df.to_csv(output_dir / "heuristic_runs.csv", index=False)
    by_seed.to_csv(output_dir / "heuristic_by_seed.csv", index=False)
    overall.to_csv(output_dir / "heuristic_overall.csv", index=False)
    plot(by_seed, overall, output_dir / "heuristic_summary.png")
    write_readable_summary(df, by_seed, overall, output_dir)

    print(f"[plot_heuristic_summary] Wrote {output_dir / 'heuristic_runs.csv'}")
    print(f"[plot_heuristic_summary] Wrote {output_dir / 'heuristic_by_seed.csv'}")
    print(f"[plot_heuristic_summary] Wrote {output_dir / 'heuristic_overall.csv'}")
    print(f"[plot_heuristic_summary] Wrote {output_dir / 'heuristic_summary.md'}")
    print(f"[plot_heuristic_summary] Wrote {output_dir / 'heuristic_summary_rounded.csv'}")
    print(f"[plot_heuristic_summary] Wrote {output_dir / 'heuristic_summary.png'}")


if __name__ == "__main__":
    main()
