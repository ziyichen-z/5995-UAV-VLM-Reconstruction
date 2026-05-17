"""
capture_manager_live.py
带实时双 viewport 刷新的 CaptureManager。

左侧 viewport：俯视正交总览，看到相机在房间里的位置
右侧 viewport：当前相机的拍摄视野

配合 setup_viewport.py 使用：
  1. 在 Blender Scripting 面板里先运行 setup_viewport.py
  2. 再用 run_live.sh 启动主程序
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
        """移动相机并同时刷新左侧总览和右侧相机视野。"""
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
        将墙体/地板对象在 viewport 中隐藏（hide_set 不影响渲染输出）。
        每次切换相机时调用，确保场景建好后墙壁保持隐藏。
        """
        keywords = ('Wall', 'wall', 'Floor', 'floor', 'Ceiling', 'ceiling')
        for obj in bpy.data.objects:
            if any(kw in obj.name for kw in keywords):
                if not obj.hide_get():          # 避免重复打印
                    obj.hide_set(True)

    def _configure_viewports(self, cam_obj):
        """
        三栏 viewport 布局（每次切换相机时调用）：
          左栏：俯视正交总览
          中栏：全视角透视（固定斜上方视角，看到房间整体结构）
          右栏：当前激活相机的拍摄视野
        """
        import mathutils

        view3d_areas = []
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    view3d_areas.append(area)

        if not view3d_areas:
            return

        # 隐藏墙体（场景已建好后生效）
        self._hide_walls_viewport()

        # 按 x 坐标排序，保证 左→中→右 顺序正确
        view3d_areas.sort(key=lambda a: a.x)

        # ── 只有 1 个 viewport：直接设为相机视野 ─────────────────────────────
        if len(view3d_areas) == 1:
            for space in view3d_areas[0].spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'CAMERA'
            return

        # ── 2 个 viewport（兼容旧布局）：左=俯视  右=相机 ────────────────────
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

        # ── 3 个及以上 viewport：左=俯视  中=全视角  右=相机 ─────────────────
        # 左栏：俯视正交
        for space in view3d_areas[0].spaces:
            if space.type == 'VIEW_3D':
                r3d = space.region_3d
                r3d.view_perspective = 'ORTHO'
                if r3d.view_distance != 20.0:   # 只在首次配置时写入，避免闪烁
                    r3d.view_rotation = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
                    r3d.view_distance = 20.0
                    r3d.view_location = (0.0, 0.0, 0.0)
                space.overlay.show_extras = True
                break

        # 中栏：全视角透视（固定斜上方 45°，不随相机切换而变化）
        for space in view3d_areas[1].spaces:
            if space.type == 'VIEW_3D':
                r3d = space.region_3d
                r3d.view_perspective = 'PERSP'
                if r3d.view_distance != 25.0:   # 只在首次配置时写入
                    r3d.view_rotation = mathutils.Euler(
                        (1.1, 0.0, -0.785), 'XYZ').to_quaternion()
                    r3d.view_distance = 25.0
                    r3d.view_location = (0.0, 0.0, 0.0)
                space.overlay.show_extras = True
                break

        # 右栏：当前激活相机视野
        for space in view3d_areas[2].spaces:
            if space.type == 'VIEW_3D':
                space.region_3d.view_perspective = 'CAMERA'
                break

        print(f"[CaptureManagerLive] 3-pane: ortho | persp | {cam_obj.name} cam")