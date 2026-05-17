"""
run_heuristic_repeats.py

Run heuristic-only supplementary capture for a small seed set, then aggregate
the heuristic results.

Default:
  python3 scripts/run_heuristic_repeats.py

Runs seeds 42, 20260427, 20260428 once each with a 3-visit heuristic budget.
Use --repeats 5 if you want five repeats per seed.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
DEFAULT_SEEDS = [42, 20260427, 20260428]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "pipeline_config.json"))
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--visits", type=int, default=3)
    parser.add_argument("--frames-per-visit", type=int, default=None)
    parser.add_argument("--fast-render", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use low-cost render settings for coverage-only heuristic benchmarks.")
    parser.add_argument("--skip-video", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_run_config(base_cfg: dict, seed: int, rep: int,
                     output_root: Path, visits: int,
                     frames_per_visit: int,
                     fast_render: bool) -> dict:
    cfg = json.loads(json.dumps(base_cfg))
    cfg["scene"]["seed"] = seed
    cfg["scene"]["mesh_folder"] = str(REPO_ROOT / "Meshy")
    cfg["scene"]["output_manifest"] = str(
        output_root / "manifests" / f"room_seed{seed}_rep{rep:02d}.json"
    )
    cfg["scene"]["manifest_path"] = cfg["scene"]["output_manifest"]
    cfg["output"]["base_dir"] = str(output_root)
    cfg["output"]["run_prefix"] = f"seed{seed}_rep{rep:02d}"

    cfg.setdefault("budget", {})
    cfg["budget"]["frames_per_visit"] = frames_per_visit
    cfg["budget"]["max_visits"] = visits
    cfg["budget"]["total_frames"] = visits * frames_per_visit

    cfg.setdefault("exploration", {})
    cfg["exploration"]["max_rounds"] = 1
    cfg["exploration"]["visits_schedule"] = [visits]

    cfg.setdefault("comparison", {})
    cfg["comparison"]["heuristic_visits_per_round"] = visits
    cfg["comparison"]["heuristic_max_candidates"] = cfg["comparison"].get(
        "heuristic_max_candidates",
        cfg.get("coverage", {}).get("max_candidates", 8),
    )
    cfg["comparison"]["heuristic_force_visit_budget"] = True

    if fast_render:
        cfg.setdefault("render", {})
        cfg["render"]["resolution_x"] = 320
        cfg["render"]["resolution_y"] = 180
        cfg["render"]["samples"] = 1

    return cfg


def newest_matching_run(output_root: Path, prefix: str, started_at: float):
    candidates = []
    for path in output_root.glob(f"{prefix}_*"):
        if path.is_dir() and path.stat().st_mtime >= started_at - 2:
            candidates.append((path.stat().st_mtime, path))
    return sorted(candidates)[-1][1] if candidates else None


def heuristic_complete(run_dir: Path) -> bool:
    path = run_dir / "comparison" / "selection_policy_comparison.json"
    if not path.exists():
        return False
    policies = load_json(path).get("policies", {})
    return "heuristic" in policies


def run_one(args, config_path: Path):
    cmd = [
        args.blender,
        "--background",
        "--python", str(REPO_ROOT / "scripts" / "run_pipeline.py"),
        "--",
        "--config", str(config_path),
        "--selection-policy", "heuristic",
    ]
    if args.skip_video:
        cmd.append("--skip-video")
    print(f"[run_heuristic_repeats] Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env).returncode


def write_manifest_line(output_root: Path, record: dict):
    with open(output_root / "heuristic_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_plot(output_root: Path, seeds: list[int]):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plot_heuristic_summary.py"),
        "--input-root", str(output_root),
        "--output-dir", str(output_root / "summary"),
        "--seeds", *[str(seed) for seed in seeds],
    ]
    print(f"[run_heuristic_repeats] Plotting: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main():
    args = parse_args()
    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "outputs" / f"heuristic_benchmark_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        run_plot(output_root, args.seeds)
        return

    base_cfg = load_json(Path(args.config))
    frames_per_visit = args.frames_per_visit or base_cfg.get("budget", {}).get("frames_per_visit", 20)
    configs_dir = output_root / "configs"
    failures = []

    for seed in args.seeds:
        for rep in range(1, args.repeats + 1):
            prefix = f"seed{seed}_rep{rep:02d}"
            existing = sorted(output_root.glob(f"{prefix}_*"))
            if existing and heuristic_complete(existing[-1]):
                print(f"[run_heuristic_repeats] Skipping complete run: {existing[-1]}")
                continue

            config_path = configs_dir / f"{prefix}.json"
            write_json(
                config_path,
                build_run_config(
                    base_cfg, seed, rep, output_root,
                    visits=args.visits,
                    frames_per_visit=frames_per_visit,
                    fast_render=args.fast_render,
                ),
            )
            started_at = time.time()
            code = run_one(args, config_path)
            run_dir = newest_matching_run(output_root, prefix, started_at)
            record = {
                "seed": seed,
                "rep": rep,
                "config": str(config_path),
                "returncode": code,
                "run_dir": str(run_dir) if run_dir else None,
            }
            write_manifest_line(output_root, record)
            if code != 0 or run_dir is None or not heuristic_complete(run_dir):
                failures.append(record)
                print(f"[run_heuristic_repeats] FAILED or incomplete: {record}")
            else:
                print(f"[run_heuristic_repeats] Complete: {run_dir}")

    if failures:
        write_json(output_root / "failed_runs.json", {"failures": failures})
        print(f"[run_heuristic_repeats] {len(failures)} run(s) failed or incomplete.")

    run_plot(output_root, args.seeds)
    print(f"[run_heuristic_repeats] Output root: {output_root}")


if __name__ == "__main__":
    main()
