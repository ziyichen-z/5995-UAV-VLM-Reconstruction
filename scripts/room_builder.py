import bpy
import json
import math
import random
import sys
from pathlib import Path
from mathutils import Vector

# ============================================================
# CONFIG LOADER
# ============================================================
# room_builder.py no longer uses a hardcoded CONFIG dict.
# Usage:
#   Standalone: blender --background --python room_builder.py -- --config /path/to/pipeline_config.json
#   As module:  from room_builder import load_config, build_scene_from_config, build_scene_from_manifest
#               cfg = load_config("/path/to/pipeline_config.json")
#               build_scene_from_config(cfg)

# Global CONFIG dict - populated by load_config()
CONFIG = {}


def _resolve_path(base_dir: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_config(config_path: str) -> dict:
    """
    Load configuration from pipeline_config.json and map it to the
    room_builder CONFIG structure. Sets the global CONFIG and returns it.
    """
    global CONFIG
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        pipeline_cfg = json.load(f)

    config_dir = path.resolve().parent
    scene = pipeline_cfg["scene"]

    # Map pipeline_config.json fields to room_builder's expected CONFIG keys
    CONFIG = {
        "seed": scene["seed"],
        "mode": scene["mode"],
        "force_complete_random": scene.get("force_complete_random", False),
        "mesh_folder": _resolve_path(config_dir, scene["mesh_folder"]),
        "output_json": _resolve_path(config_dir, scene["output_manifest"]),
        "manifest_path": _resolve_path(config_dir, scene["manifest_path"]),
        "clear_scene_first": scene.get("clear_scene_first", True),
        "origin": scene.get("origin", [0.0, 0.0, 0.0]),
        "room": scene["room"],
        "placement": scene["placement"],
        "assets": scene.get("assets", []),
    }
    return CONFIG


def _parse_cli_config() -> str:
    """Extract --config argument from Blender command line (after '--' separator)."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return argv[i + 1]
    return ""

# ============================================================
# GENERAL HELPERS
# ============================================================

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)
    for block in bpy.data.cameras:
        if block.users == 0:
            bpy.data.cameras.remove(block)
    for block in bpy.data.lights:
        if block.users == 0:
            bpy.data.lights.remove(block)

def set_seed(seed):
    random.seed(seed)

def get_glb_files(folder):
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    return sorted([str(f) for f in p.iterdir() if f.suffix.lower() == ".glb"])

def sample_uniform(rng):
    return random.uniform(rng[0], rng[1])

def room_bounds(room_cfg):
    return {
        "x": [-room_cfg["width"] / 2.0, room_cfg["width"] / 2.0],
        "y": [-room_cfg["depth"] / 2.0, room_cfg["depth"] / 2.0],
        "z": [0.0, room_cfg["height"]],
    }

def remove_accidental_ceiling():
    for name in ["Ceiling", "Roof", "Top", "Lid"]:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)

# ============================================================
# ROOM BUILDING
# ============================================================

def create_room(room_cfg):
    """
    Creates an open-top room by default:
      - floor
      - 4 walls
      - ceiling only if open_top == False
    """
    width = room_cfg["width"]
    depth = room_cfg["depth"]
    height = room_cfg["height"]
    wall_t = room_cfg["wall_thickness"]
    open_top = room_cfg.get("open_top", True)

    created = {}

    # Floor
    bpy.ops.mesh.primitive_cube_add(location=(0, 0, -wall_t / 2.0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.scale = (width / 2.0, depth / 2.0, wall_t / 2.0)
    created["floor"] = floor

    # Left wall
    bpy.ops.mesh.primitive_cube_add(location=(-(width / 2.0 + wall_t / 2.0), 0, height / 2.0))
    wall_left = bpy.context.active_object
    wall_left.name = "Wall_Left"
    wall_left.scale = (wall_t / 2.0, depth / 2.0, height / 2.0)
    created["wall_left"] = wall_left

    # Right wall
    bpy.ops.mesh.primitive_cube_add(location=((width / 2.0 + wall_t / 2.0), 0, height / 2.0))
    wall_right = bpy.context.active_object
    wall_right.name = "Wall_Right"
    wall_right.scale = (wall_t / 2.0, depth / 2.0, height / 2.0)
    created["wall_right"] = wall_right

    # Back wall
    bpy.ops.mesh.primitive_cube_add(location=(0, -(depth / 2.0 + wall_t / 2.0), height / 2.0))
    wall_back = bpy.context.active_object
    wall_back.name = "Wall_Back"
    wall_back.scale = (width / 2.0, wall_t / 2.0, height / 2.0)
    created["wall_back"] = wall_back

    # Front wall
    bpy.ops.mesh.primitive_cube_add(location=(0, (depth / 2.0 + wall_t / 2.0), height / 2.0))
    wall_front = bpy.context.active_object
    wall_front.name = "Wall_Front"
    wall_front.scale = (width / 2.0, wall_t / 2.0, height / 2.0)
    created["wall_front"] = wall_front

    # Only create a ceiling if explicitly requested
    if not open_top:
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, height + wall_t / 2.0))
        ceiling = bpy.context.active_object
        ceiling.name = "Ceiling"
        ceiling.scale = (width / 2.0, depth / 2.0, wall_t / 2.0)
        created["ceiling"] = ceiling

    remove_accidental_ceiling() if open_top else None
    return created

def resolve_room_config(config):
    room_cfg = config["room"]

    if config["mode"] == "manual":
        room_mode = "manual"
    elif config["mode"] == "random":
        room_mode = room_cfg.get("mode", "random")
    else:
        room_mode = room_cfg.get("mode", "random")

    if config.get("force_complete_random", False):
        room_mode = "random"

    if room_mode == "manual":
        resolved = dict(room_cfg["manual"])
        resolved["resolved_from_mode"] = "manual"
        resolved["open_top"] = bool(resolved.get("open_top", True))
        resolved["sources"] = {
            "width": "manual",
            "depth": "manual",
            "height": "manual",
            "wall_thickness": "manual",
            "open_top": "manual",
        }
        return resolved

    if room_mode == "random":
        rr = room_cfg["random"]
        resolved = {
            "name": f'{rr.get("name_prefix", "room")}_{random.randint(0, 999999):06d}',
            "width": sample_uniform(rr["width_range"]),
            "depth": sample_uniform(rr["depth_range"]),
            "height": sample_uniform(rr["height_range"]),
            "wall_thickness": sample_uniform(rr["wall_thickness_range"]),
            "open_top": bool(rr.get("open_top", True)),
            "resolved_from_mode": "random",
            "sources": {
                "width": "random",
                "depth": "random",
                "height": "random",
                "wall_thickness": "random",
                "open_top": "fixed_true" if rr.get("open_top", True) else "fixed_false",
            }
        }
        return resolved

    raise ValueError(f"Unsupported room mode: {room_mode}")

# ============================================================
# IMPORT / OBJECT STRUCTURE HELPERS
# ============================================================

def import_glb(filepath):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.gltf(filepath=filepath)
    after = set(bpy.data.objects.keys())
    new_names = list(after - before)
    return [bpy.data.objects[name] for name in new_names]

def get_root_objects(objects):
    obj_set = set(objects)
    roots = []
    for obj in objects:
        if obj.parent not in obj_set:
            roots.append(obj)
    return roots

def make_collection_instance_empty(name, objects):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    empty = bpy.context.active_object
    empty.name = name

    roots = get_root_objects(objects)
    for obj in roots:
        obj.parent = empty

    return empty

def all_descendants(root):
    result = []
    stack = [root]
    while stack:
        current = stack.pop()
        result.append(current)
        stack.extend(list(current.children))
    return result

def combined_world_bbox(objects):
    pts = []
    for obj in objects:
        if obj.type == 'MESH':
            pts.extend([obj.matrix_world @ Vector(corner) for corner in obj.bound_box])

    if not pts:
        locs = [obj.matrix_world.translation for obj in objects]
        if not locs:
            return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
        xs = [p.x for p in locs]
        ys = [p.y for p in locs]
        zs = [p.z for p in locs]
        return {
            "min": [min(xs), min(ys), min(zs)],
            "max": [max(xs), max(ys), max(zs)]
        }

    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    zs = [p.z for p in pts]

    return {
        "min": [min(xs), min(ys), min(zs)],
        "max": [max(xs), max(ys), max(zs)],
    }

def bbox_size(aabb):
    return [
        aabb["max"][0] - aabb["min"][0],
        aabb["max"][1] - aabb["min"][1],
        aabb["max"][2] - aabb["min"][2],
    ]

def aabb_overlap_3d(a, b, gap=0.0):
    return not (
        a["max"][0] + gap <= b["min"][0] or
        a["min"][0] >= b["max"][0] + gap or
        a["max"][1] + gap <= b["min"][1] or
        a["min"][1] >= b["max"][1] + gap or
        a["max"][2] + gap <= b["min"][2] or
        a["min"][2] >= b["max"][2] + gap
    )

def bbox_inside_room(aabb, room_cfg):
    bounds = room_bounds(room_cfg)
    return (
        aabb["min"][0] >= bounds["x"][0] and
        aabb["max"][0] <= bounds["x"][1] and
        aabb["min"][1] >= bounds["y"][0] and
        aabb["max"][1] <= bounds["y"][1] and
        aabb["min"][2] >= bounds["z"][0] and
        aabb["max"][2] <= bounds["z"][1]
    )

def shift_group_to_floor(objects):
    aabb = combined_world_bbox(objects)
    min_z = aabb["min"][2]
    dz = -min_z
    for obj in objects:
        obj.location.z += dz

def find_asset_cfg(filename):
    for asset_cfg in CONFIG.get("assets", []):
        if asset_cfg.get("file") == filename:
            return asset_cfg
    return None

def get_random_xy_ranges_from_room(room_cfg):
    margin = CONFIG["placement"]["random"].get("margin_from_walls", 0.25)
    return {
        "x_range": [-(room_cfg["width"] / 2.0) + margin, (room_cfg["width"] / 2.0) - margin],
        "y_range": [-(room_cfg["depth"] / 2.0) + margin, (room_cfg["depth"] / 2.0) - margin],
    }

# ============================================================
# TRANSFORM RESOLUTION
# ============================================================

def resolve_asset_transform(asset_cfg, room_cfg):
    global_mode = CONFIG["mode"]
    placement_mode = CONFIG["placement"].get("mode", "random")

    if CONFIG.get("force_complete_random", False):
        effective_mode = "random"
    elif asset_cfg and "mode" in asset_cfg:
        effective_mode = asset_cfg["mode"]
    elif global_mode == "manual":
        effective_mode = "manual"
    else:
        effective_mode = placement_mode

    room_ranges = get_random_xy_ranges_from_room(room_cfg)

    if effective_mode == "manual" and asset_cfg and "manual" in asset_cfg:
        m = asset_cfg["manual"]
        return {
            "mode": "manual",
            "location": m["location"],
            "rotation_deg": m["rotation_deg"],
            "scale_uniform": m["scale_uniform"],
            "sources": {
                "location": "manual",
                "rotation": "manual",
                "scale": "manual",
            }
        }

    random_cfg = {}
    if asset_cfg and "random" in asset_cfg:
        random_cfg = asset_cfg["random"]

    yaw_range = random_cfg.get("yaw_range_deg", CONFIG["placement"]["random"]["random_yaw_deg"])
    scale_range = random_cfg.get("scale_range", CONFIG["placement"]["random"]["random_scale_uniform"])
    x_range = random_cfg.get("x_range", room_ranges["x_range"])
    y_range = random_cfg.get("y_range", room_ranges["y_range"])

    return {
        "mode": "random",
        "location": [
            random.uniform(x_range[0], x_range[1]),
            random.uniform(y_range[0], y_range[1]),
            0.0
        ],
        "rotation_deg": [
            0.0,
            0.0,
            random.uniform(yaw_range[0], yaw_range[1]),
        ],
        "scale_uniform": random.uniform(scale_range[0], scale_range[1]),
        "sources": {
            "location": "random",
            "rotation": "random",
            "scale": "random",
        }
    }

def apply_transform(root_empty, transform):
    root_empty.location = tuple(transform["location"])
    root_empty.rotation_euler = tuple(math.radians(v) for v in transform["rotation_deg"])
    s = transform["scale_uniform"]
    root_empty.scale = (s, s, s)

# ============================================================
# SCENE BUILD FROM CONFIG
# ============================================================

def build_scene_from_config():
    set_seed(CONFIG["seed"])
    room_cfg = resolve_room_config(CONFIG)
    create_room(room_cfg)

    glb_files = get_glb_files(CONFIG["mesh_folder"])
    if not glb_files:
        raise RuntimeError(f"No .glb files found in: {CONFIG['mesh_folder']}")

    placed_asset_aabbs = []
    asset_records = []

    max_attempts = CONFIG["placement"]["random"]["max_attempts_per_asset"]
    min_gap = CONFIG["placement"]["random"]["min_gap"]
    z_on_floor = CONFIG["placement"]["random"]["z_on_floor"]

    for filepath in glb_files:
        imported = import_glb(filepath)
        if not imported:
            print(f"Skipping empty import: {filepath}")
            continue

        asset_name = Path(filepath).stem
        filename = Path(filepath).name
        asset_cfg = find_asset_cfg(filename)

        root_empty = make_collection_instance_empty(f"{asset_name}_ROOT", imported)
        all_objs = all_descendants(root_empty)

        if z_on_floor:
            shift_group_to_floor(all_objs)
            bpy.context.view_layer.update()

        success = False
        attempts_used = 0
        chosen_transform = None
        final_bbox = None

        target_mode = "random"
        if asset_cfg and asset_cfg.get("mode") == "manual" and not CONFIG.get("force_complete_random", False):
            target_mode = "manual"
        elif CONFIG["mode"] == "manual" and asset_cfg and "manual" in asset_cfg and not CONFIG.get("force_complete_random", False):
            target_mode = "manual"

        if target_mode == "manual":
            chosen_transform = resolve_asset_transform(asset_cfg, room_cfg)
            apply_transform(root_empty, chosen_transform)
            bpy.context.view_layer.update()

            all_objs = all_descendants(root_empty)
            final_bbox = combined_world_bbox(all_objs)

            if bbox_inside_room(final_bbox, room_cfg):
                collision = False
                for existing in placed_asset_aabbs:
                    if aabb_overlap_3d(final_bbox, existing, gap=min_gap):
                        collision = True
                        break
                if not collision:
                    success = True
                    placed_asset_aabbs.append(final_bbox)

            attempts_used = 1

        else:
            for attempt in range(max_attempts):
                attempts_used = attempt + 1

                chosen_transform = resolve_asset_transform(asset_cfg, room_cfg)
                apply_transform(root_empty, chosen_transform)
                bpy.context.view_layer.update()

                all_objs = all_descendants(root_empty)
                final_bbox = combined_world_bbox(all_objs)

                if not bbox_inside_room(final_bbox, room_cfg):
                    continue

                collision = False
                for existing in placed_asset_aabbs:
                    if aabb_overlap_3d(final_bbox, existing, gap=min_gap):
                        collision = True
                        break

                if collision:
                    continue

                success = True
                placed_asset_aabbs.append(final_bbox)
                break

        obj_names = [obj.name for obj in imported]
        bbox_dims = bbox_size(final_bbox) if final_bbox else [0.0, 0.0, 0.0]

        asset_records.append({
            "asset_name": asset_name,
            "source_file": filepath,
            "source_filename": filename,
            "imported_object_names": obj_names,
            "placement_mode": chosen_transform["mode"] if chosen_transform else None,
            "placement_success": success,
            "attempts_used": attempts_used,
            "location_world": chosen_transform["location"] if chosen_transform else None,
            "location_relative_to_origin": chosen_transform["location"] if chosen_transform else None,
            "rotation_euler_deg": chosen_transform["rotation_deg"] if chosen_transform else None,
            "scale_mode": chosen_transform["sources"]["scale"] if chosen_transform else None,
            "scale_uniform": chosen_transform["scale_uniform"] if chosen_transform else None,
            "parameter_sources": chosen_transform["sources"] if chosen_transform else None,
            "bounding_box_world": final_bbox,
            "bounding_box_size_world": bbox_dims,
            "semantic_class": asset_cfg.get("semantic_class") if asset_cfg else None,
        })

        if not success:
            print(f"WARNING: Could not place {asset_name} cleanly inside the room.")

    manifest = {
        "dataset_metadata": {
            "seed": CONFIG["seed"],
            "units": "meters",
            "mesh_folder": CONFIG["mesh_folder"],
            "coordinate_origin_world": CONFIG["origin"],
            "build_mode": CONFIG["mode"],
            "force_complete_random": CONFIG["force_complete_random"],
        },
        "room": {
            "name": room_cfg["name"],
            "width": room_cfg["width"],
            "depth": room_cfg["depth"],
            "height": room_cfg["height"],
            "wall_thickness": room_cfg["wall_thickness"],
            "open_top": room_cfg.get("open_top", True),
            "bounds": room_bounds(room_cfg),
            "resolved_from_mode": room_cfg["resolved_from_mode"],
            "sources": room_cfg["sources"],
        },
        "assets": asset_records,
        "scene_metrics": {
            "asset_count": len(asset_records),
            "successful_placements": sum(1 for a in asset_records if a["placement_success"]),
            "failed_placements": sum(1 for a in asset_records if not a["placement_success"]),
        }
    }

    outpath = Path(CONFIG["output_json"])
    outpath.parent.mkdir(parents=True, exist_ok=True)

    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Manifest written to: {outpath}")

# ============================================================
# SCENE REBUILD FROM MANIFEST
# ============================================================

def build_scene_from_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    room_cfg = {
        "name": manifest["room"]["name"],
        "width": manifest["room"]["width"],
        "depth": manifest["room"]["depth"],
        "height": manifest["room"]["height"],
        "wall_thickness": manifest["room"]["wall_thickness"],
        "open_top": manifest["room"].get("open_top", True),
    }

    create_room(room_cfg)

    for asset in manifest["assets"]:
        filepath = asset["source_file"]

        if not Path(filepath).exists():
            print(f"WARNING: Missing file during rebuild, skipping: {filepath}")
            continue

        imported = import_glb(filepath)
        if not imported:
            print(f"WARNING: Empty import during rebuild: {filepath}")
            continue

        asset_name = Path(filepath).stem
        root_empty = make_collection_instance_empty(f"{asset_name}_ROOT", imported)
        all_objs = all_descendants(root_empty)

        if CONFIG["placement"]["random"]["z_on_floor"]:
            shift_group_to_floor(all_objs)

        location = asset.get("location_world", [0.0, 0.0, 0.0])
        rotation_deg = asset.get("rotation_euler_deg", [0.0, 0.0, 0.0])
        scale_uniform = asset.get("scale_uniform", 1.0)

        root_empty.location = tuple(location)
        root_empty.rotation_euler = tuple(math.radians(v) for v in rotation_deg)
        root_empty.scale = (scale_uniform, scale_uniform, scale_uniform)

        bpy.context.view_layer.update()

    print(f"Done. Scene rebuilt from manifest: {manifest_path}")

# ============================================================
# MAIN
# ============================================================

def main():
    config_path = _parse_cli_config()
    if not config_path:
        raise RuntimeError(
            "No --config argument provided.\n"
            "Usage: blender --background --python room_builder.py -- "
            "--config /path/to/pipeline_config.json"
        )
    load_config(config_path)
    print(f"[RoomBuilder] Config loaded from: {config_path}")

    if CONFIG["clear_scene_first"]:
        clear_scene()

    if CONFIG["mode"] == "rebuild":
        build_scene_from_manifest(CONFIG["manifest_path"])
    else:
        build_scene_from_config()


if __name__ == "__main__":
    main()
