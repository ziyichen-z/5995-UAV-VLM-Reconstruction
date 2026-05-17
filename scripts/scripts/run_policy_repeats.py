"""
run_policy_repeats.py

Run repeated VLM-vs-heuristic comparisons with equal supplementary visit
budgets, then aggregate and plot the results.

Default experiment:
  - seeds: 42, 20260427, 20260428, 20260429, 20260430
  - repeats per seed: 5
  - visits per policy: 3

Example:
  python3 scripts/run_policy_repeats.py --repeats 5 --visits 3
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
DEFAULT_SEEDS = [42, 20260427, 20260428, 20260429, 20260430]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "pipeline_config.json"))
    parser.add_argument("--blender", default=DEFAULT_BLENDER)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--visits", type=int, default=3)
    parser.add_argument("--frames-per-visit", type=int, default=None)
    parser.add_argument("--skip-video", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="Only aggregate/plot an existing --output-root.")
    parser.add_argument("--allow-mock-vlm", action="store_true",
                        help="Allow runs without the configured VLM API key.")
    parser.add_argument("--max-launch-failures", type=int, default=3,
                        help="Abort after this many runs fail before creating a run dir.")
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
                     frames_per_visit: int) -> dict:
    cfg = json.loads(json.dumps(base_cfg))

    cfg["scene"]["seed"] = seed
    cfg["scene"]["mesh_folder"] = str(REPO_ROOT / "Meshy")
    cfg["scene"]["output_manifest"] = str(output_root / "manifests" / f"room_seed{seed}_rep{rep:02d}.json")
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

    return cfg


def newest_matching_run(output_root: Path, prefix: str, started_at: float):
    candidates = []
    for path in output_root.glob(f"{prefix}_*"):
        if not path.is_dir():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime >= started_at - 2:
            candidates.append((mtime, path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def comparison_complete(run_dir: Path) -> bool:
    path = run_dir / "comparison" / "selection_policy_comparison.json"
    if not path.exists():
        return False
    data = load_json(path)
    policies = data.get("policies", {})
    return "vlm" in policies and "heuristic" in policies


def run_one(args, config_path: Path):
    cmd = [
        args.blender,
        "--background",
        "--python", str(REPO_ROOT / "scripts" / "run_pipeline.py"),
        "--",
        "--config", str(config_path),
        "--selection-policy", "both",
    ]
    if args.skip_video:
        cmd.append("--skip-video")

    print(f"[run_policy_repeats] Running: {' '.join(cmd)}")
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    return proc.returncode


def write_manifest_line(output_root: Path, record: dict):
    path = output_root / "benchmark_runs.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_plot(output_root: Path):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plot_policy_comparison.py"),
        "--input-root", str(output_root),
        "--output-dir", str(output_root / "summary"),
    ]
    print(f"[run_policy_repeats] Plotting: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main():
    args = parse_args()
    if args.output_root:
        output_root = Path(args.output_root).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_root = REPO_ROOT / "outputs" / f"policy_benchmark_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        run_plot(output_root)
        return

    base_cfg = load_json(Path(args.config))
    provider = base_cfg.get("vlm", {}).get("provider", "anthropic")
    required_env = "DASHSCOPE_API_KEY" if provider == "qwen" else "ANTHROPIC_API_KEY"
    if not args.allow_mock_vlm and not os.environ.get(required_env):
        raise SystemExit(
            f"{required_env} is not set. Refusing to run a mock-VLM benchmark. "
            "Set the API key in the environment or pass --allow-mock-vlm."
        )

    frames_per_visit = args.frames_per_visit or base_cfg.get("budget", {}).get("frames_per_visit", 20)
    configs_dir = output_root / "configs"

    failures = []
    launch_failures = 0
    for seed in args.seeds:
        for rep in range(1, args.repeats + 1):
            prefix = f"seed{seed}_rep{rep:02d}"
            existing = sorted(output_root.glob(f"{prefix}_*"))
            if existing and comparison_complete(existing[-1]):
                print(f"[run_policy_repeats] Skipping complete run: {existing[-1]}")
                continue

            cfg = build_run_config(
                base_cfg, seed, rep, output_root,
                visits=args.visits,
                frames_per_visit=frames_per_visit,
            )
            config_path = configs_dir / f"{prefix}.json"
            write_json(config_path, cfg)

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

            if code != 0 or run_dir is None or not comparison_complete(run_dir):
                failures.append(record)
                print(f"[run_policy_repeats] FAILED or incomplete: {record}")
                if run_dir is None:
                    launch_failures += 1
                    if launch_failures >= args.max_launch_failures:
                        print("[run_policy_repeats] Aborting: repeated Blender launch "
                              f"failures ({launch_failures}).")
                        write_json(output_root / "failed_runs.json",
                                   {"failures": failures})
                        return
            else:
                launch_failures = 0
                print(f"[run_policy_repeats] Complete: {run_dir}")

    if failures:
        print(f"[run_policy_repeats] {len(failures)} run(s) failed or incomplete.")
        write_json(output_root / "failed_runs.json", {"failures": failures})

    if not args.dry_run:
        run_plot(output_root)
    print(f"[run_policy_repeats] Output root: {output_root}")


if __name__ == "__main__":
    main()
