import os
import sys
from pathlib import Path

import bpy
import mathutils


PROJECT_ROOT = Path(
    os.environ.get(
        "PIPE_VISUAL_ROOT",
        Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd(),
    )
).resolve()
CONFIG = PROJECT_ROOT / "pipeline_config.json"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
VENDOR_DIR = PROJECT_ROOT / ".vendor"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_PATH = OUTPUT_DIR / "run_live.log"

# API keys are intentionally not hardcoded. Set ANTHROPIC_API_KEY or
# DASHSCOPE_API_KEY in your shell/launch environment before running Blender.

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NOISE_PREFIXES = ("Warning: 1 x", "AL lib:", "ALSA lib")


class TeeLogger:
    """
    Mirror output to the Blender Scripting console and to a log file.
    Viewport refresh warnings are still written to disk, but are hidden from
    the console to keep live runs readable.
    """

    def __init__(self, path, original_stdout):
        self.file = open(path, "w", buffering=1, encoding="utf-8")
        self.terminal = original_stdout

    def write(self, msg):
        if not msg:
            return
        is_noise = any(msg.startswith(p) for p in NOISE_PREFIXES)
        self.file.write(msg)
        self.file.flush()
        if not is_noise:
            try:
                self.terminal.write(msg)
                self.terminal.flush()
            except Exception:
                pass

    def flush(self):
        self.file.flush()
        try:
            self.terminal.flush()
        except Exception:
            pass

    def fileno(self):
        return self.file.fileno()


_original_stdout = sys.stdout
_original_stderr = sys.stderr
logger = TeeLogger(LOG_PATH, _original_stdout)
sys.stdout = logger
sys.stderr = TeeLogger(LOG_PATH, _original_stderr)

print(f"[Launcher] Mirroring output to the Scripting console and {LOG_PATH}")


def _hide_walls_viewport():
    """
    Hide room shells in the viewport without affecting rendered output.
    If the scene has not been built yet, this silently does nothing.
    """
    keywords = ("Wall", "wall", "Floor", "floor", "Ceiling", "ceiling")
    hidden = []
    for obj in bpy.data.objects:
        if any(kw in obj.name for kw in keywords):
            obj.hide_set(True)
            hidden.append(obj.name)
    if hidden:
        print(f"[Launcher] Hidden viewport shell objects: {hidden}")


def setup_triple_viewport():
    view3d_areas = [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"]

    if not view3d_areas:
        print("[Launcher] No VIEW_3D area found; skipping viewport setup.")
        return

    # Split a single viewport into left overview and right working area.
    if len(view3d_areas) == 1:
        with bpy.context.temp_override(area=view3d_areas[0]):
            bpy.ops.screen.area_split(direction="VERTICAL", factor=0.33)
        view3d_areas = sorted(
            [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"],
            key=lambda a: a.x,
        )

    # Split the right working area into perspective and active-camera panes.
    if len(view3d_areas) == 2:
        with bpy.context.temp_override(area=view3d_areas[-1]):
            bpy.ops.screen.area_split(direction="VERTICAL", factor=0.5)
        view3d_areas = sorted(
            [a for a in bpy.context.screen.areas if a.type == "VIEW_3D"],
            key=lambda a: a.x,
        )

    if len(view3d_areas) < 3:
        print("[Launcher] Could not create three panes; continuing with current layout.")
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
            space.overlay.show_extras = True
            space.shading.type = "SOLID"
            print(f"[Launcher] Left pane (x={left_area.x}): top-down overview")
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
            space.overlay.show_extras = True
            space.shading.type = "SOLID"
            print(f"[Launcher] Middle pane (x={middle_area.x}): oblique perspective")
            break

    # Right pane: active camera view.
    for space in right_area.spaces:
        if space.type == "VIEW_3D":
            space.region_3d.view_perspective = "CAMERA"
            space.shading.type = "SOLID"
            print(f"[Launcher] Right pane (x={right_area.x}): active camera")
            break

    for area in view3d_areas:
        area.tag_redraw()
    try:
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)
    except Exception:
        pass

    _hide_walls_viewport()
    print("[Launcher] Three-pane viewport ready: overview | perspective | camera")


setup_triple_viewport()

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if VENDOR_DIR.exists() and str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

sys.argv = ["blender", "--", "--config", str(CONFIG)]
exec(open(SCRIPTS_DIR / "run_pipeline.py", encoding="utf-8").read())
