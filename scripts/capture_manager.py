"""
capture_manager.py
Phase 5: Executes trajectory and captures frames + pose logs.

For each camera, walks the waypoints, sets pose, renders PNG,
and writes pose_log.json + capture_log.json.
"""

import bpy
import json
import time
import math
from pathlib import Path
from mathutils import Euler, Matrix


class CaptureManager:
    def __init__(self, config: dict, output_dir: Path, camera_objects: dict):
        self.render_cfg = config["render"]
        self.captures_dir = output_dir / "captures"
        self.cameras = camera_objects          # {cam_id: bpy_obj}

    def capture_all(self, trajectory_plans: dict, round_id: str = "base"):
        """Capture frames for all cameras following their trajectory plans."""
        summaries = {}
        for cam_id, waypoints in trajectory_plans.items():
            cam_obj = self.cameras.get(cam_id)
            if cam_obj is None:
                print(f"[CaptureManager] WARNING: {cam_id} not found, skipping.")
                continue
            summary = self._capture_camera(cam_id, cam_obj, waypoints, round_id)
            summaries[cam_id] = summary
        return summaries

    # ------------------------------------------------------------------ #
    def _capture_camera(self, cam_id: str, cam_obj, waypoints: list,
                         round_id: str) -> dict:
        cam_dir = self.captures_dir / round_id / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        # Set active camera
        bpy.context.scene.camera = cam_obj

        pose_log = []
        capture_log = {"cam_id": cam_id, "round": round_id, "frames": []}
        t0 = time.time()

        for i, wp in enumerate(waypoints):
            frame_id = f"frame_{i:04d}"
            img_path = cam_dir / f"{frame_id}.png"

            # Apply pose
            self._apply_waypoint(cam_obj, wp)

            # Render
            bpy.context.scene.render.filepath = str(img_path)
            bpy.ops.render.render(write_still=True)

            # Record pose
            pose = self._extract_pose(cam_obj, i, str(img_path))
            pose_log.append(pose)
            capture_log["frames"].append({
                "frame_id": frame_id,
                "image": str(img_path),
                "waypoint_index": i
            })

        elapsed = time.time() - t0

        # Save logs
        pose_log_path = cam_dir / "pose_log.json"
        capture_log_path = cam_dir / "capture_log.json"
        with open(pose_log_path, "w") as f:
            json.dump(pose_log, f, indent=2)
        with open(capture_log_path, "w") as f:
            json.dump(capture_log, f, indent=2)

        summary = {
            "cam_id": cam_id,
            "round": round_id,
            "frames_captured": len(waypoints),
            "elapsed_sec": round(elapsed, 2),
            "pose_log": str(pose_log_path),
            "capture_log": str(capture_log_path),
            "output_dir": str(cam_dir)
        }
        print(f"[CaptureManager] {cam_id} → {len(waypoints)} frames in {elapsed:.1f}s")
        return summary

    # ------------------------------------------------------------------ #
    @staticmethod
    def _apply_waypoint(cam_obj, wp: tuple):
        """Set camera location and rotation from a waypoint tuple."""
        x, y, z, rx, ry, rz = wp
        cam_obj.location = (x, y, z)
        cam_obj.rotation_euler = Euler((rx, ry, rz), "XYZ")
        bpy.context.view_layer.update()

    @staticmethod
    def _extract_pose(cam_obj, frame_idx: int, img_path: str) -> dict:
        """Extract full pose info from camera's current matrix_world."""
        mx = cam_obj.matrix_world
        loc = cam_obj.location
        rot = cam_obj.rotation_euler

        # Build intrinsics estimate
        scene = bpy.context.scene
        cam_data = cam_obj.data
        fx = cam_data.lens / cam_data.sensor_width * scene.render.resolution_x
        fy = fx  # square pixels assumption
        cx = scene.render.resolution_x / 2
        cy = scene.render.resolution_y / 2

        return {
            "frame_index": frame_idx,
            "image_path": img_path,
            "location": {
                "x": round(loc.x, 6),
                "y": round(loc.y, 6),
                "z": round(loc.z, 6)
            },
            "rotation_euler_rad": {
                "rx": round(rot.x, 6),
                "ry": round(rot.y, 6),
                "rz": round(rot.z, 6)
            },
            "rotation_euler_deg": {
                "rx": round(math.degrees(rot.x), 4),
                "ry": round(math.degrees(rot.y), 4),
                "rz": round(math.degrees(rot.z), 4)
            },
            "matrix_world": [
                [round(mx[r][c], 8) for c in range(4)] for r in range(4)
            ],
            "intrinsics": {
                "fx": round(fx, 4),
                "fy": round(fy, 4),
                "cx": round(cx, 4),
                "cy": round(cy, 4),
                "width": scene.render.resolution_x,
                "height": scene.render.resolution_y
            },
            "timestamp": time.time()
        }
