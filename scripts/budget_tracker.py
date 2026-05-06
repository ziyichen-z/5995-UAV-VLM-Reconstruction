"""
budget_tracker.py
Tracks and enforces the VLM's resource consumption across all rounds.

Budget is defined in pipeline_config.json under "budget":
  total_frames        - hard cap on total supplementary frames across all rounds
  cameras_allowed     - which cameras the VLM is permitted to use
  frames_per_visit    - frames captured per VLM-specified target visit
  max_visits          - max number of visits the VLM can request total
"""

import json
from pathlib import Path


class BudgetTracker:
    def __init__(self, config: dict, output_dir: Path):
        self.cfg         = config.get("budget", {})
        self.output_dir  = output_dir
        self.log_path    = output_dir / "logs" / "budget_log.json"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Budget limits (read from config)
        self.total_frames_limit  = self.cfg.get("total_frames", 60)
        self.cameras_allowed     = self.cfg.get("cameras_allowed",
                                                ["camera_01", "camera_02", "camera_03"])
        self.frames_per_visit    = self.cfg.get("frames_per_visit", 20)
        self.max_visits          = self.cfg.get("max_visits", 3)

        # Running totals
        self.frames_used  = 0
        self.visits_used  = 0
        self.visit_log    = []

    # ------------------------------------------------------------------ #
    def frames_remaining(self) -> int:
        return max(0, self.total_frames_limit - self.frames_used)

    def visits_remaining(self) -> int:
        return max(0, self.max_visits - self.visits_used)

    def is_exhausted(self) -> bool:
        return self.frames_remaining() == 0 or self.visits_remaining() == 0

    def camera_allowed(self, cam_id: str) -> bool:
        return cam_id in self.cameras_allowed

    def can_afford_visit(self) -> bool:
        return (self.frames_remaining() >= self.frames_per_visit and
                self.visits_remaining() > 0)

    # ------------------------------------------------------------------ #
    def consume(self, cam_id: str, frames: int, rationale: str) -> bool:
        """
        Attempt to consume budget for one visit.
        Returns True if allowed, False if budget exceeded.
        """
        if not self.camera_allowed(cam_id):
            print(f"[BudgetTracker] BLOCKED: {cam_id} is not in cameras_allowed.")
            return False
        if frames > self.frames_remaining():
            print(f"[BudgetTracker] BLOCKED: need {frames} frames, "
                  f"only {self.frames_remaining()} remaining.")
            return False
        if self.visits_remaining() == 0:
            print(f"[BudgetTracker] BLOCKED: max visits ({self.max_visits}) reached.")
            return False

        self.frames_used += frames
        self.visits_used += 1
        self.visit_log.append({
            "visit":     self.visits_used,
            "camera":    cam_id,
            "frames":    frames,
            "rationale": rationale,
            "remaining_frames": self.frames_remaining(),
            "remaining_visits": self.visits_remaining()
        })
        self._save()
        print(f"[BudgetTracker] Visit {self.visits_used}/{self.max_visits}  "
              f"frames used: {self.frames_used}/{self.total_frames_limit}  "
              f"({self.frames_remaining()} remaining)")
        return True

    def summary(self) -> dict:
        return {
            "total_frames_limit":  self.total_frames_limit,
            "frames_used":         self.frames_used,
            "frames_remaining":    self.frames_remaining(),
            "max_visits":          self.max_visits,
            "visits_used":         self.visits_used,
            "visits_remaining":    self.visits_remaining(),
            "cameras_allowed":     self.cameras_allowed,
            "frames_per_visit":    self.frames_per_visit,
            "exhausted":           self.is_exhausted(),
            "visit_log":           self.visit_log
        }

    def print_status(self):
        print(f"[BudgetTracker] Frames: {self.frames_used}/{self.total_frames_limit}  "
              f"Visits: {self.visits_used}/{self.max_visits}  "
              f"Cameras: {self.cameras_allowed}")

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.summary(), f, indent=2)
