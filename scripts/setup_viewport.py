"""
setup_viewport.py

Run this once from Blender's Scripting panel to create a three-pane viewport:
  left: top-down orthographic overview of the room and cameras
  middle: fixed oblique perspective of the room interior
  right: active camera view

After this, run run_live.sh; the panes will refresh during capture.
"""

import bpy
import mathutils


def hide_walls_viewport():
    """
    Hide room shell objects in the viewport without affecting rendered output.
    If the scene has not been built yet, this quietly does nothing.
    """
    keywords = ("Wall", "wall", "Floor", "floor", "Ceiling", "ceiling")
    hidden = []
    for obj in bpy.data.objects:
        if any(kw in obj.name for kw in keywords):
            obj.hide_set(True)
            hidden.append(obj.name)
    if hidden:
        print(f"[setup] Hidden viewport shell objects: {hidden}")
    else:
        print("[setup] No shell objects found yet; they will be hidden during the run.")


def setup_triple_viewport():
    view3d_areas = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"]

    if not view3d_areas:
        print("[setup] No VIEW_3D area found. Switch to the Layout workspace first.")
        return

    # First split: left overview plus right working area.
    if len(view3d_areas) == 1:
        with bpy.context.temp_override(area=view3d_areas[0]):
            bpy.ops.screen.area_split(direction="VERTICAL", factor=0.33)
        view3d_areas = sorted(
            [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"],
            key=lambda a: a.x,
        )
        print("[setup] First split complete: left 33% + right 67%.")

    # Second split: perspective pane plus active-camera pane.
    if len(view3d_areas) == 2:
        with bpy.context.temp_override(area=view3d_areas[-1]):
            bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)
        view3d_areas = sorted(
            [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"],
            key=lambda a: a.x,
        )
        print("[setup] Second split complete: middle 33% + right 33%.")

    if len(view3d_areas) < 3:
        print("[setup] Three-pane split failed. Please split the viewport manually.")
        return

    left_area = view3d_areas[0]
    middle_area = view3d_areas[1]
    right_area = view3d_areas[2]

    # Left pane: top-down orthographic overview.
    for space in left_area.spaces:
        if space.type == "VIEW_3D":
            r3d = space.region_3d
            r3d.view_perspective = "ORTHO"
            r3d.view_rotation = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
            r3d.view_distance = 20.0
            r3d.view_location = (0.0, 0.0, 0.0)
            space.overlay.show_overlays = True
            space.overlay.show_extras = True
            space.shading.type = "SOLID"
            print("[setup] Left pane: top-down overview")
            break

    # Middle pane: fixed oblique perspective of the full room.
    for space in middle_area.spaces:
        if space.type == "VIEW_3D":
            r3d = space.region_3d
            r3d.view_perspective = "PERSP"
            r3d.view_rotation = mathutils.Euler(
                (1.1, 0.0, -0.785), "XYZ"
            ).to_quaternion()
            r3d.view_distance = 25.0
            r3d.view_location = (0.0, 0.0, 0.0)
            space.overlay.show_overlays = True
            space.overlay.show_extras = True
            space.shading.type = "SOLID"
            print("[setup] Middle pane: oblique perspective")
            break

    # Right pane: active camera view.
    for space in right_area.spaces:
        if space.type == "VIEW_3D":
            space.region_3d.view_perspective = "CAMERA"
            space.shading.type = "SOLID"
            print("[setup] Right pane: active camera")
            break

    for area in view3d_areas:
        area.tag_redraw()
    try:
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    except Exception:
        pass

    hide_walls_viewport()

    print("\n[setup] Three-pane layout ready.")
    print("[setup]   left=overview | middle=perspective | right=camera")
    print("[setup] You can now run run_live.sh.")


setup_triple_viewport()
