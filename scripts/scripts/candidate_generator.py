"""
candidate_generator.py

Generates ranked supplementary capture candidates from pose logs and room manifest.

Each candidate is a spatial region that needs more coverage, scored by a composite
priority metric combining four factors:

  priority = object_score     (does the region contain scene objects?)
           + coverage_score   (how few times was it seen?)
           + angle_score      (was it only seen from one direction?)
           + position_score   (is it a hard-to-reach corner or edge?)

The output is a ranked list ready to be shown to the VLM as a selection menu.
"""

import json
import math
from pathlib import Path
from typing import List, Tuple, Dict


class CandidateGenerator:

    # Weights for the composite priority score (must sum to 1.0)
    W_OBJECT   = 0.40   # highest: objects are the main reconstruction target
    W_COVERAGE = 0.30   # how unseen the area is
    W_ANGLE    = 0.20   # view direction diversity
    W_POSITION = 0.10   # corner/edge proximity

    def __init__(self, config: dict, room_manifest: dict):
        self.cov_cfg  = config["coverage"]
        self.room     = room_manifest["room"]
        self.bounds   = room_manifest["room"]["bounds"]
        self.assets   = room_manifest.get("assets", room_manifest.get("objects", []))
        self.res      = self.cov_cfg["grid_resolution"]
        self.margin   = config["trajectory"].get("boundary_margin", 0.5)

        # Derive half-FOV tangent from actual camera intrinsics
        # tan(half_fov) = (sensor_width/2) / lens_mm
        cam_cfg = config["cameras"]
        self.half_fov_tan = cam_cfg["sensor_width"] / (2 * cam_cfg["lens_mm"])

        xb = self.bounds["x"]
        yb = self.bounds["y"]
        self.nx = max(1, int(math.ceil((xb[1] - xb[0]) / self.res)))
        self.ny = max(1, int(math.ceil((yb[1] - yb[0]) / self.res)))

        # Per-cell accumulators
        self._seen_count  = [[0]   * self.ny for _ in range(self.nx)]
        self._angle_dirs  = [[set() for _ in range(self.ny)]
                              for _ in range(self.nx)]   # set of 8-direction sectors

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def generate(self, captures_dir: Path, round_prefix: str = "base",
                 max_candidates: int = 8) -> List[dict]:
        """
        Load all pose logs, build per-cell statistics, cluster uncovered areas,
        score each cluster, and return the top candidates sorted by priority.
        """
        self._reset()
        all_poses = self._load_all_pose_logs(captures_dir, round_prefix)

        if not all_poses:
            print("[CandidateGenerator] WARNING: No pose logs found.")
            return []

        # Accumulate per-cell statistics from all frames
        for cam_id, pose_log in all_poses:
            for pose in pose_log:
                self._accumulate_pose(pose)

        # Build object footprint map (which cells contain objects)
        object_cells = self._build_object_footprint()

        # Find candidate clusters
        clusters = self._cluster_low_coverage_cells(object_cells)

        # Score and sort
        scored = []
        for cluster in clusters:
            score, breakdown = self._score_cluster(cluster, object_cells)
            scored.append({
                "center_x":  round(cluster["center_x"], 3),
                "center_y":  round(cluster["center_y"], 3),
                "radius":    round(cluster["radius"], 3),
                "cell_count": cluster["cell_count"],
                "priority_score": round(score, 4),
                "score_breakdown": breakdown,
                "priority":  "high" if score >= 0.65 else
                             ("medium" if score >= 0.35 else "low"),
                "has_objects":    cluster["has_objects"],
                "object_ids":     cluster["object_ids"],
                "avg_seen_count": round(cluster["avg_seen_count"], 2),
                "avg_angle_diversity": round(cluster["avg_angle_diversity"], 2),
                "position_type":  cluster["position_type"]
            })

        scored.sort(key=lambda c: c["priority_score"], reverse=True)

        # Filter by minimum score threshold before passing to VLM.
        # The VLM only sees candidates the algorithm has already flagged
        # as worth investigating - it should not be re-ranking by score.
        min_score = 0.30
        filtered = [c for c in scored if c["priority_score"] >= min_score]
        top = filtered[:max_candidates]

        self._print_summary(top)
        return top

    def generate_from_poses(self, pose_logs: dict,
                            max_candidates: int = 8) -> List[dict]:
        """
        Like generate() but accepts a pre-loaded {cam_id: [pose]} dict
        instead of loading from disk. Used by analyze_from_poses() so that
        per-round coverage updates also carry fresh region_candidates.
        """
        self._reset()
        if not pose_logs:
            return []
        for pose_log in pose_logs.values():
            for pose in pose_log:
                self._accumulate_pose(pose)

        object_cells = self._build_object_footprint()
        clusters     = self._cluster_low_coverage_cells(object_cells)

        scored = []
        for cluster in clusters:
            score, breakdown = self._score_cluster(cluster, object_cells)
            scored.append({
                "center_x":           round(cluster["center_x"], 3),
                "center_y":           round(cluster["center_y"], 3),
                "radius":             round(cluster["radius"], 3),
                "cell_count":         cluster["cell_count"],
                "priority_score":     round(score, 4),
                "score_breakdown":    breakdown,
                "priority":           "high"   if score >= 0.65 else
                                      ("medium" if score >= 0.35 else "low"),
                "has_objects":        cluster["has_objects"],
                "object_ids":         cluster["object_ids"],
                "avg_seen_count":     round(cluster["avg_seen_count"], 2),
                "avg_angle_diversity": round(cluster["avg_angle_diversity"], 2),
                "position_type":      cluster["position_type"],
            })

        scored.sort(key=lambda c: c["priority_score"], reverse=True)
        filtered = [c for c in scored if c["priority_score"] >= 0.30]
        top = filtered[:max_candidates]
        self._print_summary(top)
        return top

    def generate_object_candidates(self) -> List[dict]:
        """
        Return candidates for objects that were seen too few times or
        only from a narrow range of angles.
        """
        candidates = []
        for obj in self.assets:
            bbox = obj.get("bounding_box_world", obj.get("bbox_world", {}))
            if not bbox:
                continue
            cx = (bbox["min"][0] + bbox["max"][0]) / 2
            cy = (bbox["min"][1] + bbox["max"][1]) / 2
            ci, cj = self._world_to_cell(cx, cy)

            seen   = self._seen_count[ci][cj]
            angles = len(self._angle_dirs[ci][cj])

            obj_id  = obj.get("asset_name", obj.get("id", "unknown"))
            obj_src = obj.get("source_filename", obj.get("source", ""))

            coverage_label = (
                "good"    if seen >= 5 and angles >= 4 else
                "partial" if seen >= 1 else
                "none"
            )

            if coverage_label != "good":
                candidates.append({
                    "id":               obj_id,
                    "source":           obj_src,
                    "center":           [round(cx, 3), round(cy, 3)],
                    "seen_count":       seen,
                    "angle_diversity":  angles,
                    "coverage":         coverage_label,
                    "priority_score":   round(
                        (1 - min(seen, 10) / 10) * 0.6 +
                        (1 - min(angles, 8) / 8) * 0.4, 4
                    )
                })

        candidates.sort(key=lambda o: o["priority_score"], reverse=True)
        return candidates

    # ------------------------------------------------------------------ #
    #  Data loading
    # ------------------------------------------------------------------ #
    def _load_all_pose_logs(self, captures_dir: Path,
                             round_prefix: str) -> List[Tuple[str, list]]:
        result = []
        target = captures_dir / round_prefix
        dirs   = list(target.iterdir()) if target.exists() else \
                 [d for d in captures_dir.iterdir() if d.is_dir()]
        for cam_dir in dirs:
            if not cam_dir.is_dir():
                continue
            pose_path = cam_dir / "pose_log.json"
            if pose_path.exists():
                with open(pose_path) as f:
                    result.append((cam_dir.name, json.load(f)))
        return result

    # ------------------------------------------------------------------ #
    #  Per-frame accumulation
    # ------------------------------------------------------------------ #
    def _accumulate_pose(self, pose: dict):
        loc = pose["location"]
        ci, cj = self._world_to_cell(loc["x"], loc["y"])

        # Estimate footprint radius on the floor
        fov_r = max(0.4, loc["z"] * self.half_fov_tan)
        r_cells = max(1, int(fov_r / self.res))

        # Determine horizontal look direction (8 sectors: N, NE, E, SE, S, SW, W, NW)
        rz_deg = pose["rotation_euler_deg"]["rz"]
        sector = int((rz_deg % 360) / 45)   # 0..7

        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                if di*di + dj*dj <= r_cells*r_cells:
                    ni, nj = ci + di, cj + dj
                    if 0 <= ni < self.nx and 0 <= nj < self.ny:
                        self._seen_count[ni][nj] += 1
                        self._angle_dirs[ni][nj].add(sector)

    # ------------------------------------------------------------------ #
    #  Object footprint
    # ------------------------------------------------------------------ #
    def _build_object_footprint(self) -> Dict[Tuple[int, int], List[str]]:
        """
        Returns a dict mapping (ci, cj) -> [object_ids] for cells that
        contain at least one scene object.
        """
        footprint: Dict[Tuple[int, int], List[str]] = {}
        for obj in self.assets:
            bbox = obj.get("bounding_box_world", obj.get("bbox_world", {}))
            if not bbox:
                continue
            obj_id = obj.get("asset_name", obj.get("id", "unknown"))
            # Mark all cells within the object's XY footprint
            x_min, x_max = bbox["min"][0], bbox["max"][0]
            y_min, y_max = bbox["min"][1], bbox["max"][1]
            ci0, cj0 = self._world_to_cell(x_min, y_min)
            ci1, cj1 = self._world_to_cell(x_max, y_max)
            for ci in range(ci0, ci1 + 1):
                for cj in range(cj0, cj1 + 1):
                    if 0 <= ci < self.nx and 0 <= cj < self.ny:
                        footprint.setdefault((ci, cj), []).append(obj_id)
        return footprint

    # ------------------------------------------------------------------ #
    #  Clustering
    # ------------------------------------------------------------------ #
    def _cluster_low_coverage_cells(self,
                                     object_cells: dict,
                                     seen_threshold: int = 2) -> List[dict]:
        """
        Identify cells with low coverage (seen_count <= seen_threshold),
        cluster nearby cells together, and compute per-cluster statistics.
        """
        # Collect low-coverage cells
        low_cells = []
        for ci in range(self.nx):
            for cj in range(self.ny):
                if self._seen_count[ci][cj] <= seen_threshold:
                    low_cells.append((ci, cj))

        if not low_cells:
            return []

        # Greedy clustering: merge cells within cluster_radius of each other
        cluster_radius_cells = max(2, int(1.5 / self.res))
        assigned = [False] * len(low_cells)
        clusters = []

        for i, (ci, cj) in enumerate(low_cells):
            if assigned[i]:
                continue
            cluster_cells = [(ci, cj)]
            assigned[i] = True
            for j, (ci2, cj2) in enumerate(low_cells):
                if not assigned[j]:
                    if (abs(ci - ci2) <= cluster_radius_cells and
                            abs(cj - cj2) <= cluster_radius_cells):
                        cluster_cells.append((ci2, cj2))
                        assigned[j] = True

            clusters.append(self._describe_cluster(cluster_cells, object_cells))

        return clusters

    def _describe_cluster(self, cells: List[Tuple[int, int]],
                           object_cells: dict) -> dict:
        """Compute geometric and statistical properties of a cell cluster."""
        xs = [self._cell_to_world(ci, cj)[0] for ci, cj in cells]
        ys = [self._cell_to_world(ci, cj)[1] for ci, cj in cells]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        max_dist = max(math.hypot(x - cx, y - cy) for x, y in zip(xs, ys))
        radius = max(self.res, max_dist + self.res)

        # Coverage statistics across all cells
        seen_counts   = [self._seen_count[ci][cj] for ci, cj in cells]
        angle_divs    = [len(self._angle_dirs[ci][cj]) for ci, cj in cells]
        avg_seen      = sum(seen_counts) / len(seen_counts)
        avg_angle_div = sum(angle_divs) / len(angle_divs)

        # Object membership
        obj_ids = set()
        for cell in cells:
            obj_ids.update(object_cells.get(cell, []))

        # Position type (corner / edge / center)
        xb = self.bounds["x"]
        yb = self.bounds["y"]
        edge_dist = min(cx - xb[0], xb[1] - cx, cy - yb[0], yb[1] - cy)
        pos_type  = ("corner" if edge_dist < 1.0 else
                     "edge"   if edge_dist < 2.0 else
                     "center")

        return {
            "center_x":          cx,
            "center_y":          cy,
            "radius":            radius,
            "cell_count":        len(cells),
            "has_objects":       bool(obj_ids),
            "object_ids":        list(obj_ids),
            "avg_seen_count":    avg_seen,
            "avg_angle_diversity": avg_angle_div,
            "position_type":     pos_type,
            "edge_distance_m":   edge_dist
        }

    # ------------------------------------------------------------------ #
    #  Scoring
    # ------------------------------------------------------------------ #
    def _score_cluster(self, cluster: dict,
                        object_cells: dict) -> Tuple[float, dict]:
        """
        Compute composite priority score in [0, 1].
        Higher = more important to capture.
        """
        # Object score: 1.0 if cluster contains objects, 0.0 otherwise
        object_score = 1.0 if cluster["has_objects"] else 0.0

        # Coverage score: based on how rarely cells were seen
        # avg_seen_count=0 -> 1.0, avg_seen_count>=5 -> 0.0
        coverage_score = max(0.0, 1.0 - cluster["avg_seen_count"] / 5.0)

        # Angle diversity score: 0 unique angles -> 1.0, 8 angles -> 0.0
        angle_score = max(0.0, 1.0 - cluster["avg_angle_diversity"] / 8.0)

        # Position score: corners > edges > center
        pos_scores = {"corner": 1.0, "edge": 0.5, "center": 0.0}
        position_score = pos_scores.get(cluster["position_type"], 0.0)

        composite = (
            self.W_OBJECT   * object_score   +
            self.W_COVERAGE * coverage_score +
            self.W_ANGLE    * angle_score    +
            self.W_POSITION * position_score
        )

        breakdown = {
            "object":   round(object_score,   3),
            "coverage": round(coverage_score, 3),
            "angle":    round(angle_score,    3),
            "position": round(position_score, 3),
            "weights":  {
                "object":   self.W_OBJECT,
                "coverage": self.W_COVERAGE,
                "angle":    self.W_ANGLE,
                "position": self.W_POSITION
            }
        }
        return composite, breakdown

    # ------------------------------------------------------------------ #
    #  Coordinate utilities
    # ------------------------------------------------------------------ #
    def _world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        xb, yb = self.bounds["x"], self.bounds["y"]
        ci = int((x - xb[0]) / self.res)
        cj = int((y - yb[0]) / self.res)
        return (max(0, min(self.nx - 1, ci)),
                max(0, min(self.ny - 1, cj)))

    def _cell_to_world(self, ci: int, cj: int) -> Tuple[float, float]:
        xb, yb = self.bounds["x"], self.bounds["y"]
        return (xb[0] + (ci + 0.5) * self.res,
                yb[0] + (cj + 0.5) * self.res)

    def _reset(self):
        self._seen_count = [[0]    * self.ny for _ in range(self.nx)]
        self._angle_dirs = [[set() for _ in range(self.ny)]
                             for _ in range(self.nx)]

    # ------------------------------------------------------------------ #
    #  Console summary
    # ------------------------------------------------------------------ #
    def _print_summary(self, candidates: List[dict]):
        print(f"[CandidateGenerator] Generated {len(candidates)} candidates:")
        for i, c in enumerate(candidates):
            obj_tag = f" [has objects: {c['object_ids']}]" if c["has_objects"] else ""
            print(f"  [{i}] ({c['center_x']:+.2f}, {c['center_y']:+.2f})m  "
                  f"score={c['priority_score']:.3f}  "
                  f"priority={c['priority']}  "
                  f"{c['position_type']}  "
                  f"seen_avg={c['avg_seen_count']:.1f}  "
                  f"angles={c['avg_angle_diversity']:.1f}/8"
                  f"{obj_tag}")
