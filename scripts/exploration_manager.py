"""
exploration_manager.py
Phase 9: Coordinates the VLM-guided supplementary capture loop.

Architecture:
  - Algorithm (coverage_manager) decides WHEN to stop
  - VLM decides WHERE to look next (purely visually)
  - BudgetTracker enforces resource limits
  - This manager translates VLM visual targets into actual waypoints

Fix log (2026-04):
  [FIX-1] Per-round visit cap: max_visits // max_rounds per round, so budget
          is distributed evenly across rounds. Round 1 can no longer consume
          all visits with low-quality early decisions.
  [FIX-2] Blocked frame tracking: frames rejected by spatial dedup are
          collected and forwarded to VLM in the next call, so it stops
          re-proposing the same reference_frames that were already rejected.
  [FIX-3] "No new frames" early-stop now defers to algorithm fallback
          instead of stopping immediately (graceful degradation).
"""

import json
import math
from pathlib import Path

from trajectory_planner import TrajectoryPlanner
from coverage_manager    import CoverageManager
from vlm_controller      import VLMController
from budget_tracker      import BudgetTracker

try:
    from capture_manager_live import CaptureManagerLive as CaptureManager
    print("[ExplorationManager] Using CaptureManagerLive for supplementary capture")
except ImportError:
    from capture_manager import CaptureManager
    print("[ExplorationManager] Using CaptureManager (standard) for supplementary capture")


class ExplorationManager:
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
        self.vlm           = VLMController(config, output_dir)
        self.budget        = BudgetTracker(config, output_dir)

        self.stop_threshold = config["exploration"]["stop_if_coverage_above"]
        self.max_rounds     = config["exploration"]["max_rounds"]

        exp_cfg = config["exploration"]
        self.sparsity_threshold    = exp_cfg.get("sparsity_threshold", 0.15)
        self.min_dist_sparse       = exp_cfg.get("min_dist_sparse", 2.5)
        self.min_dist_dense        = exp_cfg.get("min_dist_dense", 1.5)
        self.max_orbits_per_object = exp_cfg.get("max_orbits_per_object", 2)

        # Per-round visit cap schedule.
        # If "visits_schedule" is set in config, use it directly (must have
        # exactly max_rounds entries). Otherwise build a default flat schedule:
        # each round gets the same cap = max_visits // max_rounds.
        # Flat is better than increasing because later rounds have fewer fresh
        # positions available (more spatial dedup hits) so a higher cap just
        # wastes VLM budget on proposals that will be rejected.
        explicit = exp_cfg.get("visits_schedule", [])
        if explicit and len(explicit) == self.max_rounds:
            self.visits_schedule = [max(1, v) for v in explicit]
        else:
            base = max(1, self.budget.max_visits // self.max_rounds)
            self.visits_schedule = [base] * self.max_rounds
        print(f"[ExplorationManager] Per-round visit schedule: {self.visits_schedule} "
              f"(max_visits={self.budget.max_visits} / max_rounds={self.max_rounds})")

    # ------------------------------------------------------------------ #
    def explore(self, initial_coverage: dict, pose_logs: dict) -> dict:
        """
        Main loop: alternate between VLM visual inspection and coverage check.

        Args:
            initial_coverage: output from coverage_manager.analyze()
            pose_logs:        {cam_id: [pose_entries]} for keyframe selection
        """
        round_logs       = []
        step_logs        = []
        current_coverage = initial_coverage
        visit_history    = []   # executed visits, accumulates across all rounds
        orbit_counts     = {}   # object_id -> times orbited

        # [FIX-2] blocked_frames: reference_frame IDs that were proposed by VLM
        # but rejected by spatial dedup. Forwarded to VLM each round so it
        # stops re-proposing them. Never contains coordinates.
        blocked_frames: list[str] = []

        # Snapshot of base trajectory poses taken once at the start.
        # FPS keyframe selection always draws from this snapshot so VLM
        # receives a stable "big picture" overview every round, even after
        # many supplementary orbit frames accumulate in pose_logs.
        base_pose_logs = {cam_id: list(poses) for cam_id, poses in pose_logs.items()}

        # ── Sparsity detection (computed once from initial coverage) ──────
        stats     = initial_coverage.get("statistics", {})
        n_obj     = stats.get("n_object_cells", 0)
        total     = stats.get("total_cells", 1)
        sparsity  = n_obj / max(total, 1)
        is_sparse = sparsity < self.sparsity_threshold
        min_dist  = self.min_dist_sparse if is_sparse else self.min_dist_dense
        print(f"[ExplorationManager] Scene sparsity: {sparsity:.1%} "
              f"({'sparse' if is_sparse else 'dense'})  "
              f"min_dist={min_dist}m  "
              f"max_orbits_per_object={self.max_orbits_per_object}")

        for rnd in range(1, self.max_rounds + 1):
            ratio = current_coverage["coverage_ratio"]

            # Per-round visit cap from increasing schedule [3, 4, 5, ...]
            self.visits_per_round = self.visits_schedule[rnd - 1]

            # Decay min_dist 15% per round: Round1=1.5m, Round2=1.275m, Round3=1.05m
            round_min_dist = round(min_dist * (1.0 - 0.15 * (rnd - 1)), 3)

            print(f"\n[ExplorationManager] Round {rnd}/{self.max_rounds}  "
                  f"Coverage: {ratio:.1%}  "
                  f"Budget: {self.budget.frames_remaining()} frames remaining  "
                  f"Visit cap: {self.visits_per_round}  "
                  f"min_dist: {round_min_dist}m  "
                  f"Blocked: {len(blocked_frames)}")

            # ── Algorithm termination check (VLM never sees this) ─────────
            if ratio >= self.stop_threshold:
                print(f"[ExplorationManager] Coverage target met ({ratio:.1%}). Stopping.")
                break

            if self.budget.is_exhausted():
                print("[ExplorationManager] Budget exhausted. Stopping.")
                break

            # ── Prepare keyframes for VLM (images only, no coverage data) ─
            # Pass uncovered_regions so algorithm-guided hint frames can be
            # selected alongside FPS frames. VLM still receives images only.
            uncovered = current_coverage.get("uncovered_regions", [])
            keyframes = self._select_keyframes(pose_logs, uncovered,
                                               base_pose_logs=base_pose_logs)
            if not keyframes:
                print("[ExplorationManager] No keyframes available for VLM.")
                break

            # ── VLM visual inspection ──────────────────────────────────────
            # Inject visits_per_round so the prompt can tell VLM exactly how
            # many visits it should propose this round (not the global total).
            # Also pass blocked_frames [FIX-2] so VLM avoids re-proposing them.
            budget_for_vlm = {
                **self.budget.summary(),
                "visits_per_round": self.visits_per_round,
            }
            decision = self.vlm.decide(
                keyframes,
                budget_for_vlm,
                visit_history=visit_history,
                blocked_frames=blocked_frames,
            )

            if not decision.get("should_capture", False):
                reason = decision.get("stop_reason", "scene_looks_complete")
                print(f"[ExplorationManager] VLM: scene looks complete. ({reason})")
                break

            # ── Execute VLM visits ─────────────────────────────────────────
            frames_this_round    = 0
            # [FIX-1] Count visits executed this round; stop when cap is reached
            visits_this_round    = 0
            round_blocked_frames = []   # blocked this round (for logging)

            for visit in decision.get("visits", []):
                # [FIX-1] Per-round visit cap check
                if visits_this_round >= self.visits_per_round:
                    print(f"[ExplorationManager] Round {rnd} visit cap "
                          f"({self.visits_per_round}) reached — "
                          f"deferring remaining visits to next round.")
                    break

                if self.budget.is_exhausted():
                    print("[ExplorationManager] Budget exhausted mid-round.")
                    break

                cam_id    = visit.get("camera", "camera_01")
                approach  = visit.get("approach", "orbit_close")
                ref_frame = visit.get("reference_frame", "")
                problem   = visit.get("problem", "")

                # Check camera is allowed
                if not self.budget.camera_allowed(cam_id):
                    print(f"[ExplorationManager] Camera {cam_id} not allowed. Skipping.")
                    continue

                # Translate visual target to waypoints
                waypoints, cx, cy = self._resolve_visit(cam_id, approach, ref_frame, pose_logs)
                if not waypoints:
                    continue

                # Algorithm-side spatial deduplication.
                # Uses round_min_dist which decays each round, allowing visits
                # closer together in later rounds when remaining gaps are smaller.
                if self._is_too_close_to_history(cx, cy, visit_history, round_min_dist):
                    print(f"[ExplorationManager] Skipping visit — ({cx:.2f},{cy:.2f}) "
                          f"too close to previous visit (min_dist={round_min_dist}m).")
                    # [FIX-2] Record the rejected frame so VLM won't propose it again
                    if ref_frame and ref_frame not in blocked_frames:
                        blocked_frames.append(ref_frame)
                        round_blocked_frames.append(ref_frame)
                    continue

                # Orbit count limit
                obj_id = self._nearest_object(cx, cy)
                if obj_id and orbit_counts.get(obj_id, 0) >= self.max_orbits_per_object:
                    print(f"[ExplorationManager] Skipping visit — '{obj_id}' "
                          f"already orbited {orbit_counts[obj_id]} times.")
                    if ref_frame and ref_frame not in blocked_frames:
                        blocked_frames.append(ref_frame)
                        round_blocked_frames.append(ref_frame)
                    continue

                n_frames  = self.budget.frames_per_visit
                waypoints = waypoints[:n_frames]

                # Consume budget
                visit_round_id = f"round_{rnd:02d}_visit_{self.budget.visits_used + 1:02d}"
                ok = self.budget.consume(cam_id, len(waypoints), problem)
                if not ok:
                    continue

                visit_history.append({
                    "round":           rnd,
                    "camera":          cam_id,
                    "approach":        approach,
                    "reference_frame": ref_frame,
                    "problem":         problem,
                    "center_x":        cx,
                    "center_y":        cy,
                })
                if obj_id:
                    orbit_counts[obj_id] = orbit_counts.get(obj_id, 0) + 1

                coverage_before_visit = current_coverage["coverage_ratio"]

                # Proactively block base frames that would land on the same orbit
                # center next round, so VLM stops wasting proposals on locations
                # that spatial dedup would reject anyway.
                self._preblock_visited_frames(
                    cx, cy, round_min_dist, base_pose_logs, blocked_frames
                )

                # [FIX-1] Increment per-round visit counter after successful consume
                visits_this_round += 1

                # Execute capture
                cap_mgr = CaptureManager(self.config, self.output_dir, self.cameras)
                cap_mgr.captures_dir = self.supp_dir
                cap_mgr.capture_all({cam_id: waypoints}, round_id=visit_round_id)
                frames_this_round += len(waypoints)

                # Update pose_logs for next round's VLM input
                new_pose_log_path = (self.supp_dir / visit_round_id /
                                     cam_id / "pose_log.json")
                if new_pose_log_path.exists():
                    with open(new_pose_log_path) as f:
                        new_poses = json.load(f)
                    pose_logs.setdefault(cam_id, []).extend(new_poses)
                    print(f"[ExplorationManager] Added {len(new_poses)} new poses "
                          f"for {cam_id} to pose_logs.")

                    current_coverage = self.coverage_mgr.analyze_from_poses(pose_logs)
                    coverage_after_visit = current_coverage["coverage_ratio"]
                    step_log = {
                        "policy":          "vlm",
                        "step":            len(step_logs) + 1,
                        "round":           rnd,
                        "visit_id":        visit_round_id,
                        "camera":          cam_id,
                        "approach":        approach,
                        "reference_frame": ref_frame,
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
                        "problem":          problem,
                    }
                    step_logs.append(step_log)

                    step_path = self.supp_dir / "step_coverage_log.json"
                    step_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(step_path, "w") as f:
                        json.dump(step_logs, f, indent=2)

                    print(f"[ExplorationManager] Step {step_log['step']}: "
                          f"{coverage_before_visit:.1%} -> "
                          f"{coverage_after_visit:.1%} "
                          f"(gain {step_log['coverage_gain']:.1%})")

            if round_blocked_frames:
                print(f"[ExplorationManager] Round {rnd} newly blocked frames "
                      f"(will be excluded from next VLM call): {round_blocked_frames}")

            # ── Re-run coverage check including supplementary frames ───────
            all_poses_for_coverage = {}

            base_dir = self.base_run_dir / "captures" / "base"
            if base_dir.exists():
                for cam_dir in base_dir.iterdir():
                    if cam_dir.is_dir():
                        p = cam_dir / "pose_log.json"
                        if p.exists():
                            with open(p) as f:
                                all_poses_for_coverage[cam_dir.name] = json.load(f)

            for r in range(1, rnd + 1):
                for supp_round in sorted(self.supp_dir.glob(f"round_{r:02d}*")):
                    if not supp_round.is_dir():
                        continue
                    for cam_dir in supp_round.iterdir():
                        if not cam_dir.is_dir():
                            continue
                        p = cam_dir / "pose_log.json"
                        if p.exists():
                            with open(p) as f:
                                extra = json.load(f)
                            existing = all_poses_for_coverage.get(cam_dir.name, [])
                            all_poses_for_coverage[cam_dir.name] = existing + extra

            current_coverage = self.coverage_mgr.analyze_from_poses(
                all_poses_for_coverage
            )

            # ── 如果这一轮实际拍摄帧数为 0 ──────────────────────────────────
            # [FIX-3] Don't stop immediately — just log and continue to next
            # round. The visit cap may have deferred all VLM picks, and Round
            # N+1 will have a fresh VLM call with updated coverage info.
            if frames_this_round == 0:
                print(f"[ExplorationManager] Round {rnd}: no frames captured "
                      f"(all visits were blocked or capped). Continuing to next round.")
                # If this is the last round, stop to avoid infinite loop
                if rnd == self.max_rounds:
                    print("[ExplorationManager] Final round produced no frames. Stopping.")
                    break

            round_log = {
                "round":               rnd,
                "coverage_before":     ratio,
                "coverage_after":      current_coverage["coverage_ratio"],
                "frames_captured":     frames_this_round,
                "visits_executed":     visits_this_round,
                "visits_cap":          self.visits_per_round,
                "min_dist_used":       round_min_dist,
                "newly_blocked":       round_blocked_frames,
                "total_blocked":       len(blocked_frames),
                "vlm_assessment":      decision.get("overall_assessment", ""),
                "budget_remaining":    self.budget.frames_remaining()
            }
            round_logs.append(round_log)

            log_path = self.supp_dir / f"round_{rnd:02d}" / "round_log.json"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(round_log, f, indent=2)

        return {
            "final_coverage":   current_coverage,
            "rounds_completed": len(round_logs),
            "round_logs":       round_logs,
            "step_logs":        step_logs,
            "budget":           self.budget.summary()
        }

    # ------------------------------------------------------------------ #
    #  Algorithm-driven fallback round
    # ------------------------------------------------------------------ #
    def _run_algorithm_fallback(self, current_coverage: dict,
                                 visit_history: list, pose_logs: dict,
                                 min_dist: float) -> dict:
        """
        After VLM rounds complete, spend any remaining budget on the
        highest-priority region_candidates identified by the algorithm.

        Camera assignment is purely rule-based:
          - has_objects=True  → camera_01 orbit_close (multi-angle object detail)
          - has_objects=False → camera_03 orbit_high  (maximum area coverage)

        No VLM call is made. Spatial dedup still applies so we never
        revisit a location already covered by VLM rounds.
        """
        remaining_frames = self.budget.frames_remaining()
        remaining_visits = self.budget.visits_remaining()

        if remaining_frames < self.budget.frames_per_visit or remaining_visits == 0:
            return current_coverage

        ratio = current_coverage["coverage_ratio"]
        if ratio >= self.stop_threshold:
            return current_coverage

        candidates = current_coverage.get("region_candidates", [])
        if not candidates:
            print("[ExplorationManager] Fallback: no region candidates available.")
            return current_coverage

        print(f"[ExplorationManager] Algorithm fallback — "
              f"{remaining_frames} frames / {remaining_visits} visits remaining  "
              f"Coverage: {ratio:.1%}  Candidates: {len(candidates)}")

        fallback_round_id = "round_fallback"
        fallback_frames   = 0

        for candidate in candidates:
            if self.budget.is_exhausted():
                break
            if self.budget.frames_remaining() < self.budget.frames_per_visit:
                break

            cx = candidate["center_x"]
            cy = candidate["center_y"]

            # Skip if too close to any previously visited location
            if self._is_too_close_to_history(cx, cy, visit_history, min_dist):
                print(f"[ExplorationManager] Fallback: skipping ({cx:.2f},{cy:.2f}) "
                      f"— too close to previous visit.")
                continue

            # Rule-based camera selection (no VLM)
            if candidate.get("has_objects"):
                cam_id = "camera_01"
                height, radius = 1.2, 0.8
                approach_label = "orbit_close"
            else:
                cam_id = "camera_03"
                height, radius = 2.8, 1.0
                approach_label = "orbit_high"

            # Fallback to camera_01 if camera_03 not allowed
            if not self.budget.camera_allowed(cam_id):
                cam_id = "camera_01"
                height, radius = 1.2, 0.8
                approach_label = "orbit_close"

            target = {
                "center_x": cx,
                "center_y": cy,
                "height":   height,
                "radius":   radius,
            }
            n_frames  = self.budget.frames_per_visit
            waypoints = self.traj_planner.plan_supplementary(cam_id, target, n_frames)
            waypoints = waypoints[:n_frames]

            rationale = (f"algorithm fallback: {candidate['priority']} priority, "
                         f"{candidate['position_type']}, "
                         f"score={candidate['priority_score']:.3f}")
            ok = self.budget.consume(cam_id, len(waypoints), rationale)
            if not ok:
                continue

            visit_history.append({
                "round":           fallback_round_id,
                "camera":          cam_id,
                "approach":        approach_label,
                "reference_frame": "algorithm_candidate",
                "problem":         rationale,
                "center_x":        cx,
                "center_y":        cy,
            })

            print(f"[ExplorationManager] Fallback: {cam_id} {approach_label} "
                  f"at ({cx:.2f},{cy:.2f})  {rationale}")

            cap_mgr = CaptureManager(self.config, self.output_dir, self.cameras)
            cap_mgr.captures_dir = self.supp_dir
            cap_mgr.capture_all({cam_id: waypoints}, round_id=fallback_round_id)
            fallback_frames += len(waypoints)

            # Update pose_logs
            new_pose_log_path = (self.supp_dir / fallback_round_id /
                                 cam_id / "pose_log.json")
            if new_pose_log_path.exists():
                with open(new_pose_log_path) as f:
                    new_poses = json.load(f)
                pose_logs.setdefault(cam_id, []).extend(new_poses)

        if fallback_frames == 0:
            print("[ExplorationManager] Fallback: no new locations found "
                  "(all candidates too close to previous visits).")
            return current_coverage

        # Re-compute coverage with fallback frames included
        all_poses: dict = {}

        base_dir = self.base_run_dir / "captures" / "base"
        if base_dir.exists():
            for cam_dir in base_dir.iterdir():
                if cam_dir.is_dir():
                    p = cam_dir / "pose_log.json"
                    if p.exists():
                        with open(p) as f:
                            all_poses[cam_dir.name] = json.load(f)

        if self.supp_dir.exists():
            for rnd_dir in sorted(self.supp_dir.iterdir()):
                if not rnd_dir.is_dir():
                    continue
                for cam_dir in rnd_dir.iterdir():
                    if not cam_dir.is_dir():
                        continue
                    p = cam_dir / "pose_log.json"
                    if p.exists():
                        with open(p) as f:
                            extra = json.load(f)
                        existing = all_poses.get(cam_dir.name, [])
                        all_poses[cam_dir.name] = existing + extra

        updated = self.coverage_mgr.analyze_from_poses(all_poses)
        print(f"[ExplorationManager] Fallback complete: {fallback_frames} frames captured  "
              f"Coverage: {ratio:.1%} -> {updated['coverage_ratio']:.1%}")

        # Save fallback log
        log = {
            "round":            fallback_round_id,
            "coverage_before":  ratio,
            "coverage_after":   updated["coverage_ratio"],
            "frames_captured":  fallback_frames,
            "candidates_tried": len(candidates),
        }
        log_path = self.supp_dir / fallback_round_id / "fallback_log.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

        return updated

        # ------------------------------------------------------------------ #
    #  Keyframe selection for VLM
    # ------------------------------------------------------------------ #
    def _select_keyframes(self, pose_logs: dict,
                           uncovered_regions: list = None,
                           base_pose_logs: dict = None) -> list:
        """
        Hybrid keyframe selection:
          - FPS frames    (max_keyframes_per_camera - hint_per_cam per camera):
              Spatially diverse frames via farthest point sampling.
              Always drawn from base_pose_logs (base trajectory only) so VLM
              receives a stable "big picture" overview each round regardless of
              how many close-range supplementary orbit frames have accumulated.
          - Hint frames   (hint_per_cam per camera, default 1):
              For each major uncovered region, find the closest frame from the
              full pose_logs (base + supplementary). Ensures VLM sees imagery
              near algorithm-identified blind spots.

        VLM receives images only — no coverage scores, no coordinates.
        The split is an internal routing decision invisible to the model.
        """
        max_per_cam  = self.config["vlm"].get("max_keyframes_per_camera", 5)
        hint_per_cam = self.config["vlm"].get("hint_keyframes_per_camera", 1)
        fps_per_cam  = max(1, max_per_cam - hint_per_cam)
        regions      = uncovered_regions or []

        keyframes    = []
        hint_ids     = set()   # track hint frame IDs to avoid FPS duplicates

        # ── Step 1: Algorithm hint frames ────────────────────────────────
        # For each uncovered region, pick the closest frame from each camera.
        # Deduplicated across regions per camera (one hint frame per camera max).
        if regions:
            for cam_id, poses in pose_logs.items():
                if not poses:
                    continue
                selected_hints = 0
                used_frame_idx = set()

                # Sort regions by cell_count descending (largest blind spot first)
                sorted_regions = sorted(regions,
                                        key=lambda r: r.get("cell_count", 1),
                                        reverse=True)

                for region in sorted_regions:
                    if selected_hints >= hint_per_cam:
                        break
                    rx, ry = region["center_x"], region["center_y"]

                    # Find closest pose to this uncovered region
                    best_pose = None
                    best_dist = float("inf")
                    for pose in poses:
                        loc  = pose["location"]
                        dist = math.hypot(loc["x"] - rx, loc["y"] - ry)
                        if dist < best_dist and pose["frame_index"] not in used_frame_idx:
                            best_dist = dist
                            best_pose = pose

                    if best_pose is None:
                        continue

                    frame_id = f"{cam_id}_frame_{best_pose['frame_index']:04d}"
                    keyframes.append({
                        "frame_id":   frame_id,
                        "camera":     cam_id,
                        "image_path": best_pose["image_path"],
                        "_pose":      best_pose,
                        "_hint":      True,   # internal tag for logging only
                    })
                    hint_ids.add(frame_id)
                    used_frame_idx.add(best_pose["frame_index"])
                    selected_hints += 1

        # ── Step 2: FPS frames ────────────────────────────────────────────
        # Draw only from base trajectory frames (not supplementary orbit frames).
        # Supplementary close-range orbit frames can crowd out the broader scene
        # context VLM needs to locate remaining gaps each round.
        fps_source = base_pose_logs if base_pose_logs is not None else pose_logs
        for cam_id, poses in pose_logs.items():
            if not poses:
                continue
            base_for_cam = fps_source.get(cam_id, [])
            fps_pool = [p for p in base_for_cam
                        if f"{cam_id}_frame_{p['frame_index']:04d}" not in hint_ids]
            selected = self._farthest_point_sample(fps_pool, fps_per_cam)
            for pose in selected:
                keyframes.append({
                    "frame_id":   f"{cam_id}_frame_{pose['frame_index']:04d}",
                    "camera":     cam_id,
                    "image_path": pose["image_path"],
                    "_pose":      pose,
                    "_hint":      False,
                })

        n_hint = sum(1 for k in keyframes if k.get("_hint"))
        n_fps  = len(keyframes) - n_hint
        print(f"[ExplorationManager] Keyframes: {n_fps} FPS + {n_hint} hint "
              f"= {len(keyframes)} total  "
              f"(uncovered_regions={len(regions)})")

        return keyframes

    # ------------------------------------------------------------------ #
    #  Translate VLM visit to waypoints
    # ------------------------------------------------------------------ #
    def _resolve_visit(self, cam_id: str, approach: str,
                        ref_frame_id: str, pose_logs: dict) -> tuple:
        """
        Convert a VLM visual target into concrete waypoints.
        No coordinates are ever passed to or from the VLM.
        """
        ref_pose = self._find_pose_by_frame_id(ref_frame_id, pose_logs)
        if ref_pose is None:
            print(f"[ExplorationManager] Could not find reference frame: {ref_frame_id}")
            return [], None, None

        ref_loc  = ref_pose["location"]
        cam_x, cam_y = ref_loc["x"], ref_loc["y"]

        approach_params = {
            "orbit_close": {"height": 1.2, "radius": 0.6},
            "orbit_high":  {"height": 2.8, "radius": 1.0},
            "orbit_wide":  {"height": 1.5, "radius": 2.0},
        }
        params = approach_params.get(approach)
        if params is None:
            print(f"[ExplorationManager] WARNING: unknown approach '{approach}', "
                  f"falling back to orbit_close")
            params = approach_params["orbit_close"]

        # For orbit_close: center the orbit on the nearest scene object instead
        # of the reference camera's position. During diagonal/sweep baseline
        # trajectories the camera may be 1-3m from the object it's capturing —
        # a 0.6m orbit around the camera position would miss the object entirely.
        # For orbit_high / orbit_wide: camera position is appropriate (area coverage).
        if approach == "orbit_close":
            cx, cy = self._nearest_object_xy(cam_x, cam_y, max_assoc_dist=3.0)
            if (cx, cy) != (cam_x, cam_y):
                print(f"[ExplorationManager] orbit_close: shifted center "
                      f"({cam_x:.2f},{cam_y:.2f}) -> ({cx:.2f},{cy:.2f}) "
                      f"(nearest object)")
        else:
            cx, cy = cam_x, cam_y

        target = {
            "center_x": cx,
            "center_y": cy,
            "height":   params["height"],
            "radius":   params["radius"]
        }

        n_frames  = self.budget.frames_per_visit
        waypoints = self.traj_planner.plan_supplementary(cam_id, target, n_frames)

        print(f"[ExplorationManager] {cam_id} {approach} around "
              f"({cx:.2f},{cy:.2f}) ref={ref_frame_id} "
              f"-> {len(waypoints)} waypoints")
        return waypoints, cx, cy

    @staticmethod
    def _find_pose_by_frame_id(frame_id: str, pose_logs: dict):
        parts = frame_id.rsplit("_frame_", 1)
        if len(parts) != 2:
            return None
        cam_id = parts[0]
        try:
            frame_idx = int(parts[1])
        except ValueError:
            return None
        poses = pose_logs.get(cam_id, [])
        for pose in poses:
            if pose.get("frame_index") == frame_idx:
                return pose
        return None

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

    @staticmethod
    def _farthest_point_sample(poses: list, n: int) -> list:
        if not poses:
            return []
        n = min(n, len(poses))
        selected_idx = [0]

        while len(selected_idx) < n:
            max_min_dist = -1
            farthest_idx = None
            for i, pose in enumerate(poses):
                if i in selected_idx:
                    continue
                loc = pose["location"]
                min_d = min(
                    math.hypot(loc["x"] - poses[j]["location"]["x"],
                               loc["y"] - poses[j]["location"]["y"])
                    for j in selected_idx
                )
                if min_d > max_min_dist:
                    max_min_dist = min_d
                    farthest_idx = i
            if farthest_idx is None:
                break
            selected_idx.append(farthest_idx)

        return [poses[i] for i in selected_idx]

    def _preblock_visited_frames(self, cx: float, cy: float,
                                   current_min_dist: float,
                                   base_pose_logs: dict,
                                   blocked_frames: list) -> None:
        """
        After a successful visit at (cx, cy), proactively add base-frame IDs to
        blocked_frames so VLM won't re-propose locations near the visited center.

        Uses only camera positions from the drone's own pose log — no scene
        manifest or object bounding boxes required. Any base frame whose camera
        position is within current_min_dist of (cx, cy) is blocked because:
          - If proposed as orbit_high/wide: orbit center = camera position → dedup hit
          - If proposed as orbit_close: orbit shifts to nearby object, but the
            camera itself was close → reactive dedup still catches it

        This keeps the pre-blocking fully reproducible in real-world deployments
        (pose log always available; no prior scene knowledge needed).
        """
        new_blocks = []
        for cam_id, poses in base_pose_logs.items():
            for pose in poses:
                loc = pose["location"]
                fid = f"{cam_id}_frame_{pose['frame_index']:04d}"
                if fid in blocked_frames:
                    continue
                # Block frames whose camera position is within min_dist of the
                # visited center. This is purely pose-based: no scene manifest
                # needed. Any frame used as orbit_high would center here; any
                # frame used as orbit_close would shift to a nearby object but
                # still land close enough to be caught by reactive dedup anyway.
                if math.hypot(loc["x"] - cx, loc["y"] - cy) < current_min_dist:
                    new_blocks.append(fid)
                    blocked_frames.append(fid)

        n_base = sum(len(v) for v in base_pose_logs.values())
        print(f"[ExplorationManager] Pre-block scan: center=({cx:.2f},{cy:.2f}) "
              f"min_dist={current_min_dist}m  base_frames={n_base}  "
              f"new_blocks={len(new_blocks)}")

    def _nearest_object_xy(self, cx: float, cy: float,
                            max_assoc_dist: float = 3.0) -> tuple:
        """
        Return the XY center of the nearest scene object's bounding box.
        Falls back to (cx, cy) if no object is found within max_assoc_dist.
        Used to shift orbit_close trajectories onto the actual object rather
        than the reference camera's position (which may be 1-3m away).
        """
        assets = self.room_manifest.get("assets", self.room_manifest.get("objects", []))
        best_center = (cx, cy)
        best_dist   = float("inf")
        for obj in assets:
            bbox = obj.get("bounding_box_world", obj.get("bbox_world", {}))
            if not bbox:
                continue
            ocx = (bbox["min"][0] + bbox["max"][0]) / 2
            ocy = (bbox["min"][1] + bbox["max"][1]) / 2
            d   = math.hypot(cx - ocx, cy - ocy)
            if d < best_dist:
                best_dist   = d
                best_center = (ocx, ocy)
        if best_dist < max_assoc_dist:
            return best_center
        return (cx, cy)

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
