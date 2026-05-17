"""
coverage_manager.py
Phase 7: Analyzes spatial coverage from pose logs and images.

Strategy:
  1. Project each camera frustum onto a 2D floor grid.
  2. Count how many times each grid cell is "seen" across all frames.
  3. Identify uncovered / under-covered cells.
  4. Extract keyframes for VLM input.
  5. Output a structured coverage_summary.json.
"""

import json
import math
from pathlib import Path
from typing import List, Tuple, Dict


class CoverageManager:
    def __init__(self, config: dict, room_manifest: dict, output_dir: Path):
        self.config  = config
        self.cov_cfg = config["coverage"]
        self.room    = room_manifest["room"]
        self.bounds  = room_manifest["room"]["bounds"]
        self.room_manifest = room_manifest
        # Support both "assets" (room_builder format) and "objects" (legacy format)
        self.objects = room_manifest.get("assets", room_manifest.get("objects", []))
        self.res = self.cov_cfg["grid_resolution"]  # meters per cell
        self.threshold = self.cov_cfg["min_coverage_threshold"]
        self.keyframe_interval = self.cov_cfg["keyframe_interval"]
        # A cell is "covered" only if seen from >= min_angles distinct directions
        # (used for binary reporting in _find_uncovered_regions / _assess_object_coverage)
        self.min_angles = self.cov_cfg.get("min_angles", 3)
        # Non-linear exponent for occupancy score compression (0 < alpha <= 1)
        # alpha=0.5 (sqrt): strong reward for first few angles, diminishing returns after
        self.score_alpha = self.cov_cfg.get("score_alpha", 0.5)
        # Dynamic object cell weight bounds
        self.object_weight_min = self.cov_cfg.get("object_weight_min", 20)
        self.object_weight_max = self.cov_cfg.get("object_weight_max", 200)

        # Derive half-FOV tangent from actual camera intrinsics
        # tan(half_fov) = (sensor_width/2) / lens_mm
        cam_cfg = config["cameras"]
        self.half_fov_tan = cam_cfg["sensor_width"] / (2 * cam_cfg["lens_mm"])

        # Analysis output dirs
        self.analysis_dir = output_dir / "analysis"
        self.keyframes_dir = self.analysis_dir / "keyframes"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)

        # Build floor grid
        self.grid, self.grid_shape = self._init_grid()

        # Pre-compute which grid cells contain objects (XY footprint projection).
        # Object cells require min_angles directions to be considered covered.
        # Empty cells only need to be seen from >= 1 direction.
        self.object_cells = self._build_object_cells()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def analyze(self, captures_dir: Path, round_prefix: str = "base") -> dict:
        """Load all pose logs from a directory and compute coverage summary."""
        self._reset_grid()

        all_pose_logs = []
        target = captures_dir / round_prefix
        cam_dirs = list(target.iterdir()) if target.exists() else \
                   [d for d in captures_dir.iterdir() if d.is_dir()]

        for cam_dir in cam_dirs:
            if not cam_dir.is_dir():
                continue
            pose_path = cam_dir / "pose_log.json"
            if not pose_path.exists():
                continue
            with open(pose_path) as f:
                pose_log = json.load(f)
            all_pose_logs.append((cam_dir.name, pose_log))
            self._accumulate(pose_log)

        summary = self._build_summary(all_pose_logs)

        # CandidateGenerator only runs when we have a captures_dir to reference
        try:
            from candidate_generator import CandidateGenerator
            gen = CandidateGenerator(self.config, self.room_manifest)
            summary["region_candidates"] = gen.generate(
                captures_dir, round_prefix,
                max_candidates=self.cov_cfg.get("max_candidates", 8)
            )
            summary["object_candidates"] = gen.generate_object_candidates()
        except Exception as e:
            print(f"[CoverageManager] CandidateGenerator skipped: {e}")
            summary["region_candidates"] = []
            summary["object_candidates"] = []

        out_path = self.analysis_dir / "coverage_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def analyze_from_poses(self, pose_logs: dict) -> dict:
        """
        Compute coverage from a pre-assembled dict of pose logs.
        Used by exploration_manager to include both base and supplementary frames.
        Runs CandidateGenerator.generate_from_poses() so region_candidates are
        available for the algorithm fallback and per-round coverage updates.
        """
        self._reset_grid()
        all_pose_logs = []
        for cam_id, poses in pose_logs.items():
            self._accumulate(poses)
            all_pose_logs.append((cam_id, poses))

        summary = self._build_summary(all_pose_logs)

        try:
            from candidate_generator import CandidateGenerator
            gen = CandidateGenerator(self.config, self.room_manifest)
            summary["region_candidates"] = gen.generate_from_poses(
                pose_logs,
                max_candidates=self.cov_cfg.get("max_candidates", 8)
            )
            summary["object_candidates"] = gen.generate_object_candidates()
        except Exception as e:
            print(f"[CoverageManager] CandidateGenerator skipped in analyze_from_poses: {e}")
            summary["region_candidates"] = []
            summary["object_candidates"] = []

        out_path = self.analysis_dir / "coverage_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _build_summary(self, all_pose_logs: list) -> dict:

        # ── Dynamic weight ─────────────────────────────────────────────────
        # Object cells collectively equal empty cells in total weight,
        # clamped to [object_weight_min, object_weight_max].
        total_cells   = self.grid_shape[0] * self.grid_shape[1]
        n_obj_cells   = len(self.object_cells)
        n_empty_cells = total_cells - n_obj_cells

        if n_obj_cells > 0:
            natural_w = n_empty_cells / n_obj_cells
            w_object  = max(self.object_weight_min,
                            min(self.object_weight_max, natural_w))
        else:
            w_object = 1.0

        # ── Per-cell occupancy score (continuous, non-linear) ──────────────
        # raw  = angle_dirs / 8          (fraction of 8 possible sectors seen)
        # score = raw ^ alpha            (sqrt by default → diminishing returns)
        # This means revisiting the same spot from the same direction adds
        # nothing; new angles matter most early on.
        weighted_score = 0.0
        weight_total   = 0.0
        for ci in range(self.grid_shape[0]):
            for cj in range(self.grid_shape[1]):
                raw   = len(self.grid[ci][cj]) / 8.0
                score = raw ** self.score_alpha
                w     = w_object if (ci, cj) in self.object_cells else 1.0
                weighted_score += score * w
                weight_total   += w

        coverage_ratio = weighted_score / max(weight_total, 1.0)

        # Find uncovered regions (center of uncovered cells)
        uncovered = self._find_uncovered_regions()
        object_coverage = self._assess_object_coverage()
        keyframes = self._extract_keyframes(all_pose_logs)

        summary = {
            "coverage_ratio": round(coverage_ratio, 4),
            "threshold": self.threshold,
            "coverage_met": coverage_ratio >= self.threshold,
            "grid": {
                "resolution_m": self.res,
                "shape": list(self.grid_shape),
                "bounds": self.bounds
            },
            "statistics": {
                "total_cells":    total_cells,
                "n_object_cells": n_obj_cells,
                "n_empty_cells":  n_empty_cells,
                "w_object":       round(w_object, 2),
                "score_alpha":    self.score_alpha,
                "weighted_score": round(weighted_score, 2),
                "weight_total":   round(weight_total, 2)
            },
            "uncovered_regions": uncovered,
            "object_coverage": object_coverage,
            "keyframes": keyframes,
            "cameras_analyzed": [name for name, _ in all_pose_logs]
        }

        out_path = self.analysis_dir / "coverage_summary.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[CoverageManager] Coverage: {coverage_ratio:.1%} "
              f"({'OK' if summary['coverage_met'] else 'below threshold'})  "
              f"object_cells={n_obj_cells}  w_object={w_object:.1f}  "
              f"alpha={self.score_alpha}")

        return summary

    # ------------------------------------------------------------------ #
    #  Grid operations
    # ------------------------------------------------------------------ #
    def _init_grid(self) -> Tuple[list, tuple]:
        xb = self.bounds["x"]
        yb = self.bounds["y"]
        nx = max(1, int(math.ceil((xb[1] - xb[0]) / self.res)))
        ny = max(1, int(math.ceil((yb[1] - yb[0]) / self.res)))
        grid = [[set() for _ in range(ny)] for _ in range(nx)]
        return grid, (nx, ny)

    def _build_object_cells(self) -> set:
        """
        Project each asset's XY bounding box onto the grid.
        Returns a set of (ci, cj) tuples that contain at least one object.
        These cells require multi-angle coverage; empty cells only need 1 view.
        """
        cells = set()
        for obj in self.objects:
            bbox = obj.get("bounding_box_world", obj.get("bbox_world", {}))
            if not bbox:
                continue
            x_min, x_max = bbox["min"][0], bbox["max"][0]
            y_min, y_max = bbox["min"][1], bbox["max"][1]
            ci0, cj0 = self._world_to_cell(x_min, y_min)
            ci1, cj1 = self._world_to_cell(x_max, y_max)
            for ci in range(ci0, ci1 + 1):
                for cj in range(cj0, cj1 + 1):
                    cells.add((ci, cj))
        print(f"[CoverageManager] Object footprint: {len(cells)} cells "
              f"out of {self.grid_shape[0] * self.grid_shape[1]} total")
        return cells

    def _reset_grid(self):
        nx, ny = self.grid_shape
        self.grid = [[set() for _ in range(ny)] for _ in range(nx)]

    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        xb = self.bounds["x"]
        yb = self.bounds["y"]
        nx, ny = self.grid_shape
        ci = int((x - xb[0]) / self.res)
        cj = int((y - yb[0]) / self.res)
        ci = max(0, min(nx - 1, ci))
        cj = max(0, min(ny - 1, cj))
        return ci, cj

    def _cell_to_world(self, ci: int, cj: int) -> Tuple[float, float]:
        xb = self.bounds["x"]
        yb = self.bounds["y"]
        x = xb[0] + (ci + 0.5) * self.res
        y = yb[0] + (cj + 0.5) * self.res
        return x, y

    def _accumulate(self, pose_log: list):
        """Project each camera position into the grid and record view directions."""
        for pose in pose_log:
            loc = pose["location"]
            cx, cy = self._world_to_cell(loc["x"], loc["y"])

            # Horizontal look direction → one of 8 sectors (N/NE/E/SE/S/SW/W/NW)
            rz_deg = pose["rotation_euler_deg"]["rz"]
            sector = int((rz_deg % 360) / 45)

            # Footprint: camera covers a circular area on the floor
            fov_radius_m = max(0.5, loc["z"] * self.half_fov_tan)
            r_cells = max(1, int(fov_radius_m / self.res))

            for di in range(-r_cells, r_cells + 1):
                for dj in range(-r_cells, r_cells + 1):
                    if di * di + dj * dj <= r_cells * r_cells:
                        ni, nj = cx + di, cy + dj
                        nx_g, ny_g = self.grid_shape
                        if 0 <= ni < nx_g and 0 <= nj < ny_g:
                            self.grid[ni][nj].add(sector)

    def _find_uncovered_regions(self) -> list:
        """Return list of under-covered area descriptors sorted by importance."""
        nx, ny = self.grid_shape
        uncovered = []
        for ci in range(nx):
            for cj in range(ny):
                angles = len(self.grid[ci][cj])
                is_object_cell = (ci, cj) in self.object_cells
                threshold = self.min_angles if is_object_cell else 1
                if angles < threshold:
                    wx, wy = self._cell_to_world(ci, cj)
                    uncovered.append({"center_x": round(wx, 3),
                                      "center_y": round(wy, 3)})

        # Cluster nearby cells into regions
        return self._cluster_regions(uncovered, cluster_dist=self.res * 3)

    def _cluster_regions(self, points: list, cluster_dist: float) -> list:
        """Simple greedy clustering of uncovered points into regions."""
        if not points:
            return []
        clusters = []
        assigned = [False] * len(points)
        for i, p in enumerate(points):
            if assigned[i]:
                continue
            cluster = [p]
            assigned[i] = True
            for j, q in enumerate(points):
                if not assigned[j]:
                    d = math.hypot(p["center_x"] - q["center_x"],
                                   p["center_y"] - q["center_y"])
                    if d < cluster_dist:
                        cluster.append(q)
                        assigned[j] = True
            xs = [c["center_x"] for c in cluster]
            ys = [c["center_y"] for c in cluster]
            clusters.append({
                "center_x": round(sum(xs) / len(xs), 3),
                "center_y": round(sum(ys) / len(ys), 3),
                "radius": round(cluster_dist / 2, 3),
                "cell_count": len(cluster),
                "priority": "high" if len(cluster) > 5 else "medium"
            })
        clusters.sort(key=lambda c: c["cell_count"], reverse=True)
        return clusters[:10]   # Top 10 regions

    def _assess_object_coverage(self) -> list:
        """Check if each object/asset's centroid was covered."""
        results = []
        for obj in self.objects:
            # Support both field names: bounding_box_world (room_builder) / bbox_world (legacy)
            bbox = obj.get("bounding_box_world", obj.get("bbox_world", {}))
            if not bbox:
                continue
            min_v = bbox["min"]
            max_v = bbox["max"]
            cx_w = (min_v[0] + max_v[0]) / 2
            cy_w = (min_v[1] + max_v[1]) / 2
            ci, cj = self._world_to_cell(cx_w, cy_w)
            angle_count = len(self.grid[ci][cj])

            obj_id   = obj.get("asset_name", obj.get("id", "unknown"))
            obj_src  = obj.get("source_filename", obj.get("source", ""))
            results.append({
                "id":          obj_id,
                "source":      obj_src,
                "center":      [round(cx_w, 3), round(cy_w, 3)],
                "seen_count":  angle_count,
                "coverage":    "good"    if angle_count >= self.min_angles else (
                               "partial" if angle_count >= 1 else "none")
            })
        return results

    def _extract_keyframes(self, all_pose_logs: list) -> list:
        """Select representative keyframes for VLM input."""
        keyframes = []
        for cam_name, pose_log in all_pose_logs:
            interval = max(1, self.keyframe_interval)
            selected = pose_log[::interval][:5]       # max 5 per camera
            for pose in selected:
                keyframes.append({
                    "camera": cam_name,
                    "frame_index": pose["frame_index"],
                    "image_path": pose["image_path"],
                    "location": pose["location"],
                    "rotation_euler_deg": pose["rotation_euler_deg"]
                })
        return keyframes