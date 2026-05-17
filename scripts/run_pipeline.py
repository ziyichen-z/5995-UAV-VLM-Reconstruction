"""
run_pipeline.py
Master pipeline orchestrator. Reads all parameters from pipeline_config.json.

Usage:
  blender --background --python scripts/run_pipeline.py -- --config configs/pipeline_config.json

Options:
  --skip-build   Skip scene build (reuse existing .blend and room_manifest.json)
  --skip-vlm     Skip VLM-guided supplementary capture
  --skip-video   Skip video export
"""

import sys
import os
import csv
import json
import time
import argparse
import shutil
from pathlib import Path
from datetime import datetime

# Add scripts/ to sys.path so Blender can find local modules
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import bpy

# Local modules — force reload every run so edits take effect in Blender GUI mode.
# In background/CLI mode this is a no-op (modules aren't cached across processes).
import importlib
import room_builder, lighting_setup, camera_initializer, trajectory_planner
import capture_manager_live, video_export, coverage_manager, candidate_generator
import vlm_controller, exploration_manager, heuristic_exploration_manager
for _mod in [room_builder, lighting_setup, camera_initializer, trajectory_planner,
             capture_manager_live, video_export, coverage_manager, candidate_generator,
             vlm_controller, exploration_manager, heuristic_exploration_manager]:
    importlib.reload(_mod)

from lighting_setup      import LightingSetup
from camera_initializer  import CameraInitializer
from trajectory_planner  import TrajectoryPlanner
from capture_manager_live import CaptureManagerLive as CaptureManager
from video_export        import VideoExporter
from coverage_manager    import CoverageManager
from vlm_controller      import VLMController
from exploration_manager import ExplorationManager
from heuristic_exploration_manager import HeuristicExplorationManager


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--config",        default="configs/pipeline_config.json")
    p.add_argument("--skip-build",    action="store_true")
    p.add_argument("--skip-vlm",      action="store_true")
    p.add_argument("--skip-video",    action="store_true")
    p.add_argument("--skip-capture",  action="store_true",
                   help="Skip phases 1-6 and reuse frames from --run-dir")
    p.add_argument("--run-dir",       default=None,
                   help="Existing run directory to reuse when --skip-capture is set")
    p.add_argument("--selection-policy",
                   choices=["vlm", "heuristic", "both"],
                   default="vlm",
                   help="Supplementary selection policy to evaluate")
    return p.parse_args(argv)


def load_pipeline_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    print(f"[Pipeline] Config loaded: {path}")
    return cfg


def make_run_dir(cfg: dict) -> Path:
    base      = Path(cfg["output"]["base_dir"])
    prefix    = cfg["output"]["run_prefix"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = base / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Pipeline] Output dir: {run_dir}")
    return run_dir


def load_room_manifest(manifest_path: str) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def load_base_pose_logs(run_dir: Path, camera_objects: dict) -> dict:
    pose_logs = {}
    for cam_id in camera_objects:
        pose_path = run_dir / "captures" / "base" / cam_id / "pose_log.json"
        if pose_path.exists():
            with open(pose_path) as f:
                pose_logs[cam_id] = json.load(f)
            print(f"[Pipeline] Loaded {len(pose_logs[cam_id])} poses for {cam_id}")
    return pose_logs


def clone_pose_logs(pose_logs: dict) -> dict:
    return json.loads(json.dumps(pose_logs))


def run_supplementary_policy(policy: str, cfg: dict, run_dir: Path,
                             camera_objects: dict, room_manifest: dict,
                             coverage_summary: dict,
                             pose_logs: dict,
                             isolated_output: bool = False) -> dict:
    policy_output_dir = run_dir / "comparison" / policy if isolated_output else run_dir

    if policy == "vlm":
        print("\n=== Phases 8-9: VLM-Guided Supplementary Capture ===")
        explorer = ExplorationManager(
            cfg, policy_output_dir, camera_objects, room_manifest,
            base_run_dir=run_dir
        )
    elif policy == "heuristic":
        print("\n=== Phases 8-9: Heuristic Supplementary Capture ===")
        explorer = HeuristicExplorationManager(
            cfg, policy_output_dir, camera_objects, room_manifest,
            base_run_dir=run_dir
        )
    else:
        raise ValueError(f"Unknown selection policy: {policy}")

    result = explorer.explore(coverage_summary, clone_pose_logs(pose_logs))
    result["policy"] = policy
    result["output_dir"] = str(policy_output_dir)
    return result


def write_selection_comparison_outputs(comparison: dict, comparison_dir: Path):
    comparison_dir.mkdir(parents=True, exist_ok=True)
    with open(comparison_dir / "selection_policy_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    rows = []
    for policy, summary in comparison.get("policies", {}).items():
        for step in summary.get("step_logs", []):
            rows.append({
                "policy":           policy,
                "step":             step.get("step"),
                "round":            step.get("round"),
                "visit_id":         step.get("visit_id"),
                "camera":           step.get("camera"),
                "approach":         step.get("approach"),
                "frames_captured":  step.get("frames_captured"),
                "coverage_before":  step.get("coverage_before"),
                "coverage_after":   step.get("coverage_after"),
                "coverage_gain":    step.get("coverage_gain"),
                "cumulative_gain":  step.get("cumulative_gain"),
                "budget_remaining": step.get("budget_remaining"),
                "output_dir":       summary.get("output_dir"),
                "problem":          step.get("problem", ""),
            })

    if not rows:
        return

    csv_path = comparison_dir / "selection_policy_steps.csv"
    fieldnames = [
        "policy", "step", "round", "visit_id", "camera", "approach",
        "frames_captured", "coverage_before", "coverage_after",
        "coverage_gain", "cumulative_gain", "budget_remaining",
        "output_dir", "problem",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_policy_result(result: dict, initial_coverage: dict) -> dict:
    final_ratio = result["final_coverage"]["coverage_ratio"]
    budget = result.get("budget", {})
    return {
        "final_coverage": final_ratio,
        "coverage_gain": round(
            final_ratio - initial_coverage["coverage_ratio"], 4
        ),
        "rounds_completed": result.get("rounds_completed", 0),
        "frames_used": budget.get("frames_used", 0),
        "visits_used": budget.get("visits_used", 0),
        "output_dir": result.get("output_dir"),
        "round_logs": result.get("round_logs", []),
        "step_logs": result.get("step_logs", []),
    }


def run_selection_comparison(cfg: dict, run_dir: Path, camera_objects: dict,
                             room_manifest: dict, coverage_summary: dict,
                             selection_policy: str) -> dict:
    pose_logs = load_base_pose_logs(run_dir, camera_objects)
    if not pose_logs:
        print("[Pipeline] No base pose logs available for supplementary capture.")
        return {}

    policies = ["vlm", "heuristic"] if selection_policy == "both" else [selection_policy]
    results = {}
    for policy in policies:
        t = time.time()
        result = run_supplementary_policy(
            policy, cfg, run_dir, camera_objects, room_manifest,
            coverage_summary, pose_logs, isolated_output=(selection_policy == "both")
        )
        summary = summarize_policy_result(result, coverage_summary)
        summary["elapsed"] = round(time.time() - t, 2)
        results[policy] = summary

    comparison = {
        "initial_coverage": coverage_summary["coverage_ratio"],
        "policies": results,
    }
    write_selection_comparison_outputs(comparison, run_dir / "comparison")

    if selection_policy == "both":
        print("\n=== Selection Policy Comparison ===")
        for policy, summary in comparison["policies"].items():
            print(f"[Pipeline] {policy}: "
                  f"{summary['final_coverage']:.1%} coverage "
                  f"(gain {summary['coverage_gain']:.1%}, "
                  f"{summary['frames_used']} frames)")
    else:
        summary = comparison["policies"][selection_policy]
        print(f"[Pipeline] Final coverage ({selection_policy}): "
              f"{summary['final_coverage']:.1%}")

    return comparison


def main():
    args    = parse_args()
    cfg     = load_pipeline_config(args.config)
    t_start = time.time()

    # ── Skip-capture mode: reuse an existing run directory ───────────────────
    if args.skip_capture:
        if not args.run_dir:
            raise ValueError("--skip-capture requires --run-dir <path/to/run_YYYYMMDD_HHMMSS>")
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        print(f"[Pipeline] Skip-capture mode. Reusing: {run_dir}")

        # Load room manifest from existing run
        manifest_path = run_dir / "manifests" / "room_manifest.json"
        if not manifest_path.exists():
            manifest_path = Path(cfg["scene"]["output_manifest"])
        room_manifest = load_room_manifest(str(manifest_path))

        # Rebuild camera objects (needed for exploration_manager)
        cam_init       = CameraInitializer(cfg, run_dir)
        cam_manifest   = cam_init.initialize()
        camera_objects = cam_init.cameras

        # Run coverage analysis on existing captures
        print("\n=== Phase 7: Coverage Analysis (existing frames) ===")
        coverage_mgr     = CoverageManager(cfg, room_manifest, run_dir)
        coverage_summary = coverage_mgr.analyze(run_dir / "captures", round_prefix="base")
        print(f"[Pipeline] Coverage: {coverage_summary['coverage_ratio']:.1%}")

        # Run supplementary selection loop
        final_coverage = coverage_summary
        if not args.skip_vlm:
            comparison = run_selection_comparison(
                cfg, run_dir, camera_objects, room_manifest,
                coverage_summary, args.selection_policy
            )
            if comparison.get("policies"):
                best = max(
                    comparison["policies"].values(),
                    key=lambda p: p["final_coverage"]
                )
                final_coverage = {"coverage_ratio": best["final_coverage"]}
        else:
            print("[Pipeline] Supplementary exploration skipped.")

        bpy.ops.wm.save_as_mainfile(filepath=str(run_dir / "scene_final.blend"))
        print(f"\n[Pipeline] Done. Output: {run_dir}")
        return

    # ── Normal full pipeline ──────────────────────────────────────────────────
    run_dir = make_run_dir(cfg)

    # Copy config to run directory for reproducibility
    (run_dir / "manifests").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "pipeline_config.json")

    pipeline_log = {
        "started": datetime.now().isoformat(),
        "config":  args.config,
        "phases":  {}
    }

    # ── Phase 1: Scene Build ──────────────────────────────────────────────────
    manifest_path = cfg["scene"]["output_manifest"]

    if not args.skip_build:
        print("\n=== Phase 1: Scene Build (room_builder) ===")
        t = time.time()

        room_builder.load_config(args.config)

        if room_builder.CONFIG["clear_scene_first"]:
            room_builder.clear_scene()

        if room_builder.CONFIG["mode"] == "rebuild":
            room_builder.build_scene_from_manifest(room_builder.CONFIG["manifest_path"])
        else:
            room_builder.build_scene_from_config()

        pipeline_log["phases"]["scene_build"] = {
            "manifest": manifest_path,
            "elapsed":  round(time.time() - t, 2)
        }
    else:
        print("[Pipeline] Scene build skipped.")

    # Load room manifest written by room_builder
    room_manifest = load_room_manifest(manifest_path)

    # Copy manifest into run directory
    shutil.copy(manifest_path, run_dir / "manifests" / "room_manifest.json")

    # ── Phase 2: Lighting ─────────────────────────────────────────────────────
    print("\n=== Phase 2: Lighting ===")
    t = time.time()
    lighting = LightingSetup(cfg, room_manifest)
    lighting.setup()
    pipeline_log["phases"]["lighting"] = {"elapsed": round(time.time() - t, 2)}

    # ── Phase 3: Camera Init ──────────────────────────────────────────────────
    print("\n=== Phase 3: Camera Init ===")
    t = time.time()
    cam_init       = CameraInitializer(cfg, run_dir)
    cam_manifest   = cam_init.initialize()
    camera_objects = cam_init.cameras      # {cam_id: bpy_obj}
    pipeline_log["phases"]["camera_init"] = {
        "cameras": list(camera_objects.keys()),
        "elapsed": round(time.time() - t, 2)
    }

    # ── Phase 4: Trajectory Planning ─────────────────────────────────────────
    print("\n=== Phase 4: Trajectory Planning ===")
    t = time.time()
    planner          = TrajectoryPlanner(cfg, room_manifest)
    trajectory_plans = planner.plan_all()
    total_wpts       = sum(len(v) for v in trajectory_plans.values())
    pipeline_log["phases"]["trajectory"] = {
        "cameras":         list(trajectory_plans.keys()),
        "total_waypoints": total_wpts,
        "elapsed":         round(time.time() - t, 2)
    }

    # ── Phase 5: Baseline Capture ─────────────────────────────────────────────
    print("\n=== Phase 5: Baseline Capture ===")
    t = time.time()
    capture_mgr       = CaptureManager(cfg, run_dir, camera_objects)
    capture_summaries = capture_mgr.capture_all(trajectory_plans, round_id="base")
    total_frames      = sum(s.get("frames_captured", 0) for s in capture_summaries.values())
    pipeline_log["phases"]["baseline_capture"] = {
        "total_frames": total_frames,
        "elapsed":      round(time.time() - t, 2)
    }

    # ── Phase 6: Video Export ─────────────────────────────────────────────────
    if not args.skip_video:
        print("\n=== Phase 6: Video Export ===")
        t = time.time()
        exporter      = VideoExporter(cfg, run_dir)
        video_results = exporter.export_all(capture_summaries, round_id="base")
        video_paths   = [r["video_path"] for r in video_results.values() if r["success"]]
        if len(video_paths) > 1:
            exporter.export_overview(video_paths)
        pipeline_log["phases"]["video_export"] = {
            "videos":  video_paths,
            "elapsed": round(time.time() - t, 2)
        }

    # ── Phase 7: Coverage Analysis ────────────────────────────────────────────
    print("\n=== Phase 7: Coverage Analysis ===")
    t = time.time()
    coverage_mgr     = CoverageManager(cfg, room_manifest, run_dir)
    coverage_summary = coverage_mgr.analyze(run_dir / "captures", round_prefix="base")
    pipeline_log["phases"]["coverage_analysis"] = {
        "coverage_ratio": coverage_summary["coverage_ratio"],
        "coverage_met":   coverage_summary["coverage_met"],
        "elapsed":        round(time.time() - t, 2)
    }
    print(f"[Pipeline] Initial coverage: {coverage_summary['coverage_ratio']:.1%}")

    # ── Phases 8-9: Supplementary Capture ────────────────────────────────────
    final_coverage = coverage_summary
    if not args.skip_vlm:
        print(f"\n=== Supplementary Capture Policy: {args.selection_policy} ===")
        t = time.time()

        comparison = run_selection_comparison(
            cfg, run_dir, camera_objects, room_manifest,
            coverage_summary, args.selection_policy
        )
        if comparison.get("policies"):
            best = max(
                comparison["policies"].values(),
                key=lambda p: p["final_coverage"]
            )
            final_coverage = {"coverage_ratio": best["final_coverage"]}

        pipeline_log["phases"]["exploration"] = {
            "selection_policy": args.selection_policy,
            "initial_coverage": coverage_summary["coverage_ratio"],
            "final_coverage":   final_coverage["coverage_ratio"],
            "comparison":       comparison.get("policies", {}),
            "elapsed":          round(time.time() - t, 2)
        }
        print(f"[Pipeline] Final coverage: {final_coverage['coverage_ratio']:.1%}")
    else:
        print("[Pipeline] Supplementary exploration skipped.")

    # ── Save final .blend and archive logs ───────────────────────────────────
    bpy.ops.wm.save_as_mainfile(filepath=str(run_dir / "scene_final.blend"))

    total_elapsed = round(time.time() - t_start, 2)
    pipeline_log.update({
        "finished":             datetime.now().isoformat(),
        "total_elapsed_sec":    total_elapsed,
        "final_coverage_ratio": final_coverage.get("coverage_ratio", 0)
    })
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    with open(logs_dir / "run_summary.json", "w") as f:
        json.dump(pipeline_log, f, indent=2)

    print(f"""
+------------------------------------------+
|           Pipeline Complete              |
+------------------------------------------+
  Output:   {str(run_dir)}
  Time:     {total_elapsed:.1f}s
  Coverage: {final_coverage.get('coverage_ratio', 0):.1%}
+------------------------------------------+
""")


if __name__ == "__main__":
    main()
