"""
setup_viewport.py
在 Blender Scripting 面板里运行一次，配置好三栏 viewport 布局：
  左栏：俯视正交总览（看到整个房间和相机位置）
  中栏：全视角透视（斜上方固定视角，看到房间立体结构，墙壁已隐藏）
  右栏：当前激活相机的拍摄视野

之后再跑 run_live.sh，三个 viewport 会同步刷新。
"""

import bpy
import math
import mathutils


def hide_walls_viewport():
    """
    将墙体/地板对象在 viewport 中隐藏（不影响渲染输出）。
    如果场景尚未建好（对象不存在），静默跳过。
    """
    keywords = ('Wall', 'wall', 'Floor', 'floor', 'Ceiling', 'ceiling')
    hidden = []
    for obj in bpy.data.objects:
        if any(kw in obj.name for kw in keywords):
            obj.hide_set(True)   # 等同于 H 键：viewport 隐藏，渲染不受影响
            hidden.append(obj.name)
    if hidden:
        print(f"[setup] Viewport 隐藏墙体/地板: {hidden}")
    else:
        print("[setup] 暂无墙体对象（场景尚未建好），运行管线后自动生效。")


def setup_triple_viewport():
    # ── 找到所有现有的 VIEW_3D area ──────────────────────────────────────────
    view3d_areas = [a for a in bpy.context.screen.areas if a.type == 'VIEW_3D']

    if not view3d_areas:
        print("[setup] 找不到 VIEW_3D，请先切换到 Layout 工作区再运行。")
        return

    # ── 第一刀：把现有（或最大的）VIEW_3D 分成 左(33%) + 右(67%) ────────────
    if len(view3d_areas) == 1:
        with bpy.context.temp_override(area=view3d_areas[0]):
            bpy.ops.screen.area_split(direction='VERTICAL', factor=0.33)
        view3d_areas = sorted(
            [a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'],
            key=lambda a: a.x)
        print("[setup] 第一刀分割完成（左33% + 右67%）")

    # ── 第二刀：把最右侧 area 再纵向对半分 → 中(全视角) + 右(相机) ──────────
    if len(view3d_areas) == 2:
        with bpy.context.temp_override(area=view3d_areas[-1]):
            bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)
        view3d_areas = sorted(
            [a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'],
            key=lambda a: a.x)
        print("[setup] 第二刀分割完成（中33% + 右33%）")

    if len(view3d_areas) < 3:
        print("[setup] 三栏分割失败，请手动在 Blender 里再分割一次 viewport。")
        return

    left_area   = view3d_areas[0]   # 俯视正交
    middle_area = view3d_areas[1]   # 全视角透视
    right_area  = view3d_areas[2]   # 相机视野

    # ── 左栏：俯视正交总览 ────────────────────────────────────────────────────
    for space in left_area.spaces:
        if space.type == 'VIEW_3D':
            r3d = space.region_3d
            r3d.view_perspective = 'ORTHO'
            # 纯俯视：四元数恒等 = 从 +Z 往下看
            r3d.view_rotation   = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))
            r3d.view_distance   = 20.0
            r3d.view_location   = (0.0, 0.0, 0.0)
            space.overlay.show_overlays = True
            space.overlay.show_extras   = True   # 显示相机/灯光图标
            space.shading.type          = 'SOLID'
            print("[setup] 左栏：俯视正交总览 ✓")
            break

    # ── 中栏：全视角透视（右前上方 ~63° 俯角，隐藏墙体后可看到完整内部）──────
    for space in middle_area.spaces:
        if space.type == 'VIEW_3D':
            r3d = space.region_3d
            r3d.view_perspective = 'PERSP'
            # Euler(1.1 rad ≈ 63°, 0, -0.785 rad ≈ -45°) XYZ → 右前上方斜视
            r3d.view_rotation   = mathutils.Euler(
                (1.1, 0.0, -0.785), 'XYZ').to_quaternion()
            r3d.view_distance   = 25.0
            r3d.view_location   = (0.0, 0.0, 0.0)
            space.overlay.show_overlays = True
            space.overlay.show_extras   = True
            space.shading.type          = 'SOLID'
            print("[setup] 中栏：全视角透视 ✓")
            break

    # ── 右栏：激活相机透视视野 ────────────────────────────────────────────────
    for space in right_area.spaces:
        if space.type == 'VIEW_3D':
            space.region_3d.view_perspective = 'CAMERA'
            space.shading.type               = 'SOLID'
            print("[setup] 右栏：相机透视视野 ✓")
            break

    # 强制刷新
    for area in view3d_areas:
        area.tag_redraw()
    try:
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    except Exception:
        pass

    # ── 隐藏墙体 ─────────────────────────────────────────────────────────────
    hide_walls_viewport()

    print("\n[setup] 三栏布局配置完成。")
    print("[setup]   左=俯视总览 | 中=全视角透视（墙壁已隐藏） | 右=相机视野")
    print("[setup] 现在可以运行 run_live.sh 了。")


setup_triple_viewport()