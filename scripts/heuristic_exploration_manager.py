"""
heuristic_exploration_manager.py

Non-VLM supplementary capture policy for comparison experiments.

This manager uses CandidateGenerator scores directly:
  1. Rank under-covered regions from the current pose logs.
  2. Pick the highest-scored regions subject to budget and de-duplication.
  3. Capture supplementary orbits and recompute coverage.
"""

import json
import math
from pathlib import Path

from budget_tracker       import BudgetTracker
from candidate_generator  import CandidateGenerator
from coverage_manager     import CoverageManager
from trajectory_planner   import TrajectoryPlanner

try:
    from capture_manager_live import CaptureManagerLive as CaptureManager
    print("[HeuristicExplorationManager] Using CaptureManagerLive for supplementary capture")
except ImportError:
    from capture_manager import CaptureManager
    print("[HeuristicExplorationManager] Using CaptureManager (standard) for supplementary capture")


class HeuristicExplorationManager:
    def __init__(self, config: dict, output_dir: Path,
                 camera_objects: dict, room_manifest: dict,
                 base_run_dir: Path = None):
        self.config        = config
        self.output_dir    = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_run_dir  = base_run_dir or output_dir
        self.cameras       = camera_objects
        self.room_manifest = room_manifest
        self.supp_dir      = output_dir / "supplementary"

        self.traj_planner  = TrajectoryPlanner(config, room_manifest)
        self.coverage_mgr  = CoverageManager(config, room_manifest, output_dir)
        self.budget        = BudgetTracker(config, output_dir)

        exp_cfg = config["exploration"]
        self.stop_threshold = exp_cfg["stop_if_coverage_above"]
        self.max_rounds     = exp_cfg["max_rounds"]
        self.sparsity_threshold    = exp_cfg.get("sparsity_threshold", 0.15)
        self.min_dist_sparse       = exp_cfg.get("min_dist_sparse", 2.5)
        self.min_dist_dense        = exp_cfg.get("min_dist_dense", 1.5)
        self.max_orbits_per_object = exp_cfg.get("max_orbits_per_object", 2)

        cmp_cfg = config.get("comparison", {})
        self.visits_per_round = cmp_cfg.get("heuristic_visits_per_round", 1)
        self.max_candidates   = cmp_cfg.get("heuristic_max_candidates",
                                            config["coverage"].get("max_candidates", 8))
        self.force_visit_budget = cmp_cfg.get("heuristic_force_visit_budget", True)

    # ------------------------------------------------------------------ #
    def explore(self, initial_coverage: dict, pose_logs: dict) -> dict:
        round_logs       = []
        step_logs        = []
        current_coverage = initial_coverage
        active_pose_logs = {cam: list(poses) for cam, poses in pose_logs.items()}
        visit_history    = []
        orbit_counts     = {}

        stats     = initial_coverage.get("statistics", {})
        n_obj     = stats.get("n_object_cells", 0)
        total     = stats.get("total_cells", 1)
        sparsity  = n_obj / max(total, 1)
        is_sparse = sparsity < self.sparsity_threshold
        min_dist  = self.min_dist_sparse if is_sparse else self.min_dist_dense

        print(f"[HeuristicExplorationManager] Scene sparsity: {sparsity:.1%} "
              f"({'sparse' if is_sparse else 'dense'})  "
              f"min_dist={min_dist}m")

        for rnd in range(1, self.max_rounds + 1):
            ratio = current_coverage["coverage_ratio"]
            print(f"\n[HeuristicExplorationManager] Round {rnd}/{self.max_rounds}  "
                  f"Coverage: {ratio:.1%}  "
                  f"Budget: {self.budget.frames_remaining()} frames remaining")

            if ratio >= self.stop_threshold:
                print(f"[HeuristicExplorationManager] Coverage target met ({ratio:.1%}). Stopping.")
                break
            if self.budget.is_exhausted():
                print("[HeuristicExplorationManager] Budget exhausted. Stopping.")
                break

            candidate_generator = CandidateGenerator(self.config, self.room_manifest)
            candidates = candidate_generator.generate_from_poses(
                active_pose_logs, self.max_candidates
            )
            if not candidates and self.force_visit_budget:
                candidates = self._fallback_candidates(candidate_generator)
            if not candidates:
                print("[HeuristicExplorationManager] No heuristic candidates found.")
                break

            frames_this_round = 0
            visits_this_round = 0
            tried_relaxed_pass = False
            candidate_queue = list(candidates)
            while candidate_queue:
                candidate = candidate_queue.pop(0)
                if visits_this_round >= self.visits_per_round:
                    break
                if self.budget.is_exhausted():
                    break

                cx = candidate["center_x"]
                cy = candidate["center_y"]
                if (not tried_relaxed_pass and
                        self._is_too_close_to_history(cx, cy, visit_history, min_dist)):
                    print(f"[HeuristicExplorationManager] Skipping candidate — "
                          f"({cx:.2f},{cy:.2f}) is too close to history.")
                    continue

                obj_id = self._nearest_object(cx, cy)
                if (not tried_relaxed_pass and obj_id and
                        orbit_counts.get(obj_id, 0) >= self.max_orbits_per_object):
                    print(f"[HeuristicExplorationManager] Skipping candidate — "
                          f"'{obj_id}' already orbited {orbit_counts[obj_id]} times.")
                    continue

                cam_id, approach = self._select_camera_and_approach(candidate)
                if not self.budget.camera_allowed(cam_id):
                    continue

                waypoints = self._plan_visit(cam_id, approach, cx, cy)
                if not waypoints:
                    continue

                rationale = (
                    f"Heuristic candidate score={candidate['priority_score']} "
                    f"objects={candidate['object_ids']} "
                    f"seen={candidate['avg_seen_count']} "
                    f"angles={candidate['avg_angle_diversity']}"
                )
                visit_round_id = f"round_{rnd:02d}_visit_{self.budget.visits_used + 1:02d}"
                if not self.budget.consume(cam_id, len(waypoints), rationale):
                    continue

                visit_history.append({
                    "round": rnd,
                    "camera": cam_id,
                    "approach": approach,
                    "problem": rationale,
                    "center_x": cx,
                    "center_y": cy,
                    "candidate": candidate,
                })
                if obj_id:
                    orbit_counts[obj_id] = orbit_counts.get(obj_id, 0) + 1

                coverage_before_visit = current_coverage["coverage_ratio"]

                cap_mgr = CaptureManager(self.config, self.output_dir, self.cameras)
                cap_mgr.captures_dir = self.supp_dir
                cap_mgr.capture_all({cam_id: waypoints}, round_id=visit_round_id)
                frames_this_round += len(waypoints)
                visits_this_round += 1

                new_pose_path = self.supp_dir / visit_round_id / cam_id / "pose_log.json"
                if new_pose_path.exists():
                    with open(new_pose_path) as f:
                        new_poses = json.load(f)
                    active_pose_logs.setdefault(cam_id, []).extend(new_poses)
                    print(f"[HeuristicExplorationManager] Added {len(new_poses)} new poses "
                          f"for {cam_id} to pose logs.")

                    current_coverage = self.coverage_mgr.analyze_from_poses(active_pose_logs)
                    coverage_after_visit = current_coverage["coverage_ratio"]
                    step_log = {
                        "policy":          "heuristic",
                        "step":            len(step_logs) + 1,
                        "round":           rnd,
                        "visit_id":        visit_round_id,
                        "camera":          cam_id,
                        "approach":        approach,
                        "frames_captured": len(waypoints),
                        "coverage_before": coverage_before_visit,
                        "coverage_after":  coverage_after_visit,
                        "coverage_gain":   round(
                            coverage_after_visit - coverage_before_visit, 4
                        ),
                        "cumulative_gain": round(
                            coverage_after_visit -
                            initial_coverage["coverage_ratio"], 4
                        ),
                        "budget_remaining": self.budget.frames_remaining(),
                        "candidate":        candidate,
                        "problem":          rationale,
                    }
                    step_logs.append(step_log)

                    step_path = self.supp_dir / "step_coverage_log.json"
                    step_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(step_path, "w") as f:
                        json.dump(step_logs, f, indent=2)

                    print(f"[HeuristicExplorationManager] Step {step_log['step']}: "
                          f"{coverage_before_visit:.1%} -> "
                          f"{coverage_after_visit:.1%} "
                          f"(gain {step_log['coverage_gain']:.1%})")

                if (self.force_visit_budget and
                        not candidate_queue and
                        visits_this_round < self.visits_per_round and
                        not self.budget.is_exhausted()):
                    if not tried_relaxed_pass:
                        tried_relaxed_pass = True
                        candidate_queue = list(candidates)
                        print("[HeuristicExplorationManager] Relaxing candidate spacing "
                              "filters to use the requested visit budget.")
                    else:
                        fallback = self._fallback_candidates(candidate_generator)
                        candidate_queue = fallback
                        candidates = fallback
                        print("[HeuristicExplorationManager] Using room/object fallback "
                              "candidates to use the requested visit budget.")

            current_coverage = self.coverage_mgr.analyze_from_poses(active_pose_logs)

            if frames_this_round == 0:
                print("[HeuristicExplorationManager] No new locations captured. Stopping.")
                break

            round_log = {
                "round":            rnd,
                "coverage_before":  ratio,
                "coverage_after":   current_coverage["coverage_ratio"],
                "frames_captured":  frames_this_round,
                "visits_captured":  visits_this_round,
                "budget_remaining": self.budget.frames_remaining(),
            }
            round_logs.append(round_log)

            log_path = self.supp_dir / f"round_{rnd:02d}_summary.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(round_log, f, indent=2)

        return {
            "final_coverage":   current_coverage,
            "rounds_completed": len(round_logs),
            "round_logs":       round_logs,
            "step_logs":        step_logs,
            "budget":           self.budget.summary(),
            "visit_history":    visit_history,
        }

    # ------------------------------------------------------------------ #
    def _select_camera_and_approach(self, candidate: dict) -> tuple:
        if candidate.get("has_objects"):
            return "camera_01", "orbit_close"
        if candidate.get("position_type") in {"corner", "edge"}:
            return "camera_03", "orbit_wide"
        return "camera_03", "orbit_high"

    def _plan_visit(self, cam_id: str, approach: str, cx: float, cy: float) -> list:
        params_by_approach = {
            "orbit_close": {"height": 1.2, "radius": 0.6},
            "orbit_high":  {"height": 2.8, "radius": 1.0},
            "orbit_wide":  {"height": 1.5, "radius": 2.0},
        }
        params = params_by_approach[approach]
        target = {
            "center_x": cx,
            "center_y": cy,
            "height":   params["height"],
            "radius":   params["radius"],
        }
        waypoints = self.traj_planner.plan_supplementary(
            cam_id, target, self.budget.frames_per_visit
        )
        print(f"[HeuristicExplorationManager] {cam_id} {approach} around "
              f"({cx:.2f},{cy:.2f}) → {len(waypoints)} waypoints")
        return waypoints

    @staticmethod
    def _is_too_close_to_history(cx: float, cy: float,
                                  visit_history: list,
                                  min_dist: float) -> bool:
        for v in visit_history:
            hx = v.get("center_x")
            hy = v.get("center_y")
            if hx is None or hy is None:
                continue
            if math.hypot(cx - hx, cy - hy) < min_dist:
                return True
        return False

    def _nearest_object(self, cx: float, cy: float,
                         max_assoc_dist: float = 2.0):
        assets = self.room_manifest.get("assets", self.room_manifest.get("objects", []))
        best_id   = None
        best_dist = float("inf")
        for obj in assets:
            bbox = obj.get("bounding_box_world", obj.get("bbox_world", {}))
            if not bbox:
                continue
            ocx = (bbox["min"][0] + bbox["max"][0]) / 2
            ocy = (bbox["min"][1] + bbox["max"][1]) / 2
            d   = math.hypot(cx - ocx, cy - ocy)
            if d < best_dist:
                best_dist = d
                best_id   = obj.get("asset_name", obj.get("id", "unknown"))
        return best_id if best_dist < max_assoc_dist else None

    def _fallback_candidates(self, candidate_generator: CandidateGenerator) -> list:
        """Create deterministic fallback candidates when the strict score filter is empty."""
        candidates = []
        for obj in candidate_generator.generate_object_candidates():
            cx, cy = obj.get("center", [None, None])
            if cx is None or cy is None:
                continue
            candidates.append({
                "center_x": cx,
                "center_y": cy,
                "radius": 0.8,
                "cell_count": 1,
                "priority_score": obj.get("priority_score", 0.25),
                "score_breakdown": {"fallback": "object_candidate"},
                "priority": "fallback",
                "has_objects": True,
                "object_ids": [obj.get("id", "unknown")],
                "avg_seen_count": obj.get("seen_count", 0),
                "avg_angle_diversity": obj.get("angle_diversity", 0),
                "position_type": "center",
            })

        xb = self.room_manifest["room"]["bounds"]["x"]
        yb = self.room_manifest["room"]["bounds"]["y"]
        margin = self.config["trajectory"].get("boundary_margin", 0.5)
        room_points = [
            (xb[0] + margin, yb[0] + margin, "corner"),
            (xb[1] - margin, yb[0] + margin, "corner"),
            (xb[1] - margin, yb[1] - margin, "corner"),
            (xb[0] + margin, yb[1] - margin, "corner"),
            ((xb[0] + xb[1]) / 2, (yb[0] + yb[1]) / 2, "center"),
        ]
        for cx, cy, position_type in room_points:
            candidates.append({
                "center_x": round(cx, 3),
                "center_y": round(cy, 3),
                "radius": 1.0,
                "cell_count": 1,
                "priority_score": 0.2,
                "score_breakdown": {"fallback": "room_geometry"},
                "priority": "fallback",
                "has_objects": False,
                "object_ids": [],
                "avg_seen_count": 0,
                "avg_angle_diversity": 0,
                "position_type": position_type,
            })

        deduped = []
        seen = set()
        for candidate in candidates:
            key = (round(candidate["center_x"], 2), round(candidate["center_y"], 2))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(candidate)
        return deduped[:max(self.max_candidates, self.visits_per_round)]
