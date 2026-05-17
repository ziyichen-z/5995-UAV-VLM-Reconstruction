"""
camera_initializer.py
Phase 3: Creates the initial multi-drone camera setup.
"""

import bpy
import json
import math
from pathlib import Path
from mathutils import Matrix, Euler


class CameraInitializer:
    def __init__(self, config: dict, output_dir: Path):
        self.cfg = config["cameras"]
        self.render_cfg = config["render"]
        self.manifests_dir = output_dir / "manifests"
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.cameras: dict = {}   # id → bpy object

    def initialize(self) -> dict:
        """Create all cameras and return camera_manifest."""
        self._configure_render()
        self._remove_existing_cameras()

        camera_entries = []
        # Support both "drones" (new format) and "roles" (legacy format)
        drone_list = self.cfg.get("drones", self.cfg.get("roles", []))
        for role_cfg in drone_list:
            cam_obj = self._create_camera(role_cfg)
            self.cameras[role_cfg["id"]] = cam_obj
            camera_entries.append(self._describe(role_cfg, cam_obj))

        manifest = {"cameras": camera_entries}
        path = self.manifests_dir / "camera_manifest.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[CameraInitializer] {len(camera_entries)} cameras created → {path}")
        return manifest

    # ------------------------------------------------------------------ #
    def _configure_render(self):
        scene = bpy.context.scene
        scene.render.engine = self.render_cfg["engine"]
        scene.render.resolution_x = self.render_cfg["resolution_x"]
        scene.render.resolution_y = self.render_cfg["resolution_y"]
        scene.render.image_settings.file_format = self.render_cfg["file_format"]
        scene.render.image_settings.color_depth = self.render_cfg["color_depth"]
        if self.render_cfg["engine"] == "CYCLES":
            scene.cycles.samples = self.render_cfg["samples"]

    def _remove_existing_cameras(self):
        for obj in list(bpy.data.objects):
            if obj.type == "CAMERA":
                bpy.data.objects.remove(obj, do_unlink=True)

    def _create_camera(self, role_cfg: dict):
        # Camera data
        cam_data = bpy.data.cameras.new(name=role_cfg["id"])
        cam_data.lens = self.cfg["lens_mm"]
        cam_data.sensor_width = self.cfg["sensor_width"]
        cam_data.clip_start = self.cfg["clip_start"]
        cam_data.clip_end = self.cfg["clip_end"]

        # Camera object
        cam_obj = bpy.data.objects.new(name=role_cfg["id"], object_data=cam_data)
        bpy.context.collection.objects.link(cam_obj)

        # Set initial pose
        cam_obj.location = role_cfg["init_pos"]
        rx, ry, rz = [math.radians(d) for d in role_cfg["init_rot_deg"]]
        cam_obj.rotation_euler = Euler((rx, ry, rz), "XYZ")

        return cam_obj

    @staticmethod
    def _describe(role_cfg: dict, cam_obj) -> dict:
        mx = cam_obj.matrix_world
        return {
            "id": role_cfg["id"],
            "role": role_cfg["role"],
            "init_pos": role_cfg["init_pos"],
            "init_rot_deg": role_cfg["init_rot_deg"],
            "matrix_world": [list(row) for row in mx]
        }

    def get_camera(self, cam_id: str):
        """Return bpy camera object by id."""
        return self.cameras.get(cam_id)
