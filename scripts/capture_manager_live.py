"""
capture_manager_live.py
CaptureManager with live viewport refresh support.

Left viewport: top-down orthographic overview of camera positions.
Right viewport: active camera view.

Typical use:
  1. Run setup_viewport.py from Blender's Scripting panel.
  2. Start the main pipeline with run_live.sh.
"""

import bpy
import json
import time
from pathlib import Path
from mathutils import Euler

from capture_manager import CaptureManager


def _is_gui_mode() -> bool:
    return not bpy.app.background


class CaptureManagerLive(CaptureManager):

    def __init__(self, config: dict, output_dir: Path,
                 camera_objects: dict, viewport_delay: float = 0.0):
        super().__init__(config, output_dir, camera_objects)
        self.viewport_delay = viewport_delay
        self.gui_mode = _is_gui_mode()

        if self.gui_mode:
            print("[CaptureManagerLive] GUI mode — dual viewport refresh enabled")
        else:
            print("[CaptureManagerLive] Background mode — standard behavior")

    # ------------------------------------------------------------------ #
    def _capture_camera(self, cam_id: str, cam_obj, waypoints: list,
                        round_id: str) -> dict:
        cam_dir = self.captures_dir / round_id / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        bpy.context.scene.camera = cam_obj

        if self.gui_mode:
            self._configure_viewports(cam_obj)

        pose_log = []
        capture_log = {"cam_id": cam_id, "round": round_id, "frames": []}
        t0 = time.time()

        for i, wp in enumerate(waypoints):
            frame_id = f"frame_{i:04d}"
            img_path  = cam_dir / f"{frame_id}.png"

            self._apply_waypoint_live(cam_obj, wp, frame_number=i)

            if self.gui_mode and self.viewport_delay > 0:
                time.sleep(self.viewport_delay)

            bpy.context.scene.render.filepath = str(img_path)
            bpy.ops.render.render(write_still=True)

            pose = self._extract_pose(cam_obj, i, str(img_path))
            pose_log.append(pose)
            capture_log["frames"].append({
                "frame_id": frame_id,
                "image":    str(img_path),
                "waypoint_index": i
            })

        elapsed = time.time() - t0

        pose_log_path    = cam_dir / "pose_log.json"
        capture_log_path = cam_dir / "capture_log.json"
        with open(pose_log_path, "w") as f:
            json.dump(pose_log, f, indent=2)
        with open(capture_log_path, "w") as f:
            json.dump(capture_log, f, indent=2)

        summary = {
            "cam_id":          cam_id,
            "round":           round_id,
            "frames_captured": len(waypoints),
            "elapsed_sec":     round(elapsed, 2),
            "pose_log":        str(pose_log_path),
            "capture_log":     str(capture_log_path),
            "output_dir":      str(cam_dir)
        }
        print(f"[CaptureManagerLive] {cam_id} → {len(waypoints)} frames in {elapsed:.1f}s")
        return summary

    # ------------------------------------------------------------------ #
    def _apply_waypoint_live(self, cam_obj, wp: tuple, frame_number: int = 0):
        """Move the camera and refresh overview/camera viewports."""
        x, y, z, rx, ry, rz = wp
        cam_obj.location       = (x, y, z)
        cam_obj.rotation_euler = Euler((rx, ry, rz), "XYZ")
        bpy.context.view_layer.update()

        if not self.gui_mode:
            return

        bpy.context.scene.frame_set(frame_number)

        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

        try:
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    @staticmethod
    def _hide_walls_viewport():
        """
        Hide wall/floor objects in the viewport only.
        hide_set does not affect rendered output. This is called after camera
        switches so shell objects stay hidden once the scene exists.
        """
        keywords = ('Wall', 'wall', 'Floor', 'floor', 'Ceiling', 'ceiling')
        for obj in bpy.data.objects:
            if any(kw in obj.name for kw in keywords):
                if not obj.hide_get():
                    obj.hide_set(True)

    def _configure_viewports(self, cam_obj):
        """
        Three-pane viewport layout, refreshed on camera switches:
          left: top-down orthographic overview
          middle: fixed oblique perspective of the full room
          right: active camera view
        """
        import mathutils

        view3d_areas = []
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    view3d_areas.append(area)

        if not view3d_areas:
            return

        # Hide room shell objects after the scene has been built.
        self._hide_walls_viewport()

        # Sort by x coordinate to preserve left-to-right pane order.
        view3d_areas.sort(key=lambda a: a.x)

        # One viewport: use it as the active camera view.
        if len(view3d_areas) == 1:
            for space in view3d_areas[0].spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'CAMERA'
            return

        # Two viewports: left overview, right active camera.
        if len(view3d_areas) == 2:
            for space in view3d_areas[0].spaces:
                if space.type == 'VIEW_3D':
                    r3d = space.region_3d
                    r3d.view_perspective = 'ORTHO'
                    if r3d.view_rotation != mathutils.Quaternion((1.0, 0.0, 0.0, 0.0)):
                        r3d.view_rotation = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
                        r3d.view_distance = 20.0
                        r3d.view_location = (0.0, 0.0, 0.0)
                    space.overlay.show_extras = True
                    break
            for space in view3d_areas[1].spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'CAMERA'
                    break
            print(f"[CaptureManagerLive] 2-pane: left=ortho  right={cam_obj.name}")
            return

        # Three or more viewports: overview | perspective | camera.
        # Left pane: top-down orthographic view.
        for space in view3d_areas[0].spaces:
            if space.type == 'VIEW_3D':
                r3d = space.region_3d
                r3d.view_perspective = 'ORTHO'
                if r3d.view_distance != 20.0:
                    r3d.view_rotation = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
                    r3d.view_distance = 20.0
                    r3d.view_location = (0.0, 0.0, 0.0)
                space.overlay.show_extras = True
                break

        # Middle pane: fixed oblique perspective, independent of camera changes.
        for space in view3d_areas[1].spaces:
            if space.type == 'VIEW_3D':
                r3d = space.region_3d
                r3d.view_perspective = 'PERSP'
                if r3d.view_distance != 25.0:
                    r3d.view_rotation = mathutils.Euler(
                        (1.1, 0.0, -0.785), 'XYZ').to_quaternion()
                    r3d.view_distance = 25.0
                    r3d.view_location = (0.0, 0.0, 0.0)
                space.overlay.show_extras = True
                break

        # Right pane: active camera view.
        for space in view3d_areas[2].spaces:
            if space.type == 'VIEW_3D':
                space.region_3d.view_perspective = 'CAMERA'
                break

        print(f"[CaptureManagerLive] 3-pane: ortho | persp | {cam_obj.name} cam")
