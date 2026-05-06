# Indoor VLM-Guided Capture Pipeline

This repository contains a Blender-based indoor capture pipeline for testing
VLM-guided supplementary view planning. It builds a random room, places GLB
assets, captures a fixed multi-camera baseline trajectory, estimates coverage,
and asks a vision-language model to select additional capture visits.

## Pipeline Overview

1. Build or rebuild an indoor scene from `pipeline_config.json`.
2. Add adaptive lighting for small and large rooms.
3. Initialize three virtual drone cameras.
4. Capture a fixed baseline trajectory:
   - `camera_01`: low diagonal sweep
   - `camera_02`: low diagonal sweep from the opposite side
   - `camera_03`: high orbit for global context
5. Analyze grid/object coverage.
6. Send selected observation frames to a VLM.
7. Execute supplementary visits proposed by the VLM.
8. Recompute final coverage and save the final `.blend` scene.

The VLM is not a blind planner. The system uses heuristic coverage analysis to
select visual evidence frames, then the VLM prioritizes visually meaningful
gaps such as dark regions, missing corners, occlusions, and objects seen from
only one side.

## Repository Layout

```text
.
├── pipeline_config.json          # Main configuration
├── run_live.sh                   # Launch Blender GUI pipeline from shell
├── run_live_from_blender.py      # Launch from Blender's Scripting panel
├── requirements.txt              # Python API client dependencies
├── Meshy/                        # Place GLB assets here
└── scripts/
    ├── run_pipeline.py           # Main orchestrator
    ├── room_builder.py           # Random/manual room and asset placement
    ├── lighting_setup.py         # Adaptive lighting
    ├── trajectory_planner.py     # Baseline camera paths
    ├── coverage_manager.py       # Coverage scoring
    ├── exploration_manager.py    # VLM-guided supplementary capture loop
    └── vlm_controller.py         # Anthropic/Qwen VLM clients and prompts
```

Generated runs are written to `outputs/` and are ignored by Git by default.

## Requirements

- Blender 5.x with Python 3.11
- GLB assets placed in `Meshy/`
- Optional: `ffmpeg` for video export
- One VLM API key:
  - `ANTHROPIC_API_KEY` for Claude
  - `DASHSCOPE_API_KEY` for Qwen/DashScope

Install Python client dependencies into Blender's local vendor directory:

```bash
BLENDER_PY="/Applications/Blender.app/Contents/Resources/5.0/python/bin/python3.11"
"$BLENDER_PY" -m pip install -t .vendor -r requirements.txt
```

If Blender's Python does not have `pip`, initialize it first:

```bash
BLENDER_PY="/Applications/Blender.app/Contents/Resources/5.0/python/bin/python3.11"
"$BLENDER_PY" -m ensurepip --upgrade
```

## Configuration

Most experiment settings live in `pipeline_config.json`.

Important fields:

- `scene.seed`: random seed for room and asset placement.
- `scene.mesh_folder`: GLB asset directory, relative paths are supported.
- `output.base_dir`: generated run output directory.
- `vlm.provider`: `anthropic` or `qwen`.
- `vlm.model`: model name used by the selected provider.
- `budget.total_frames`: supplementary capture frame budget.
- `budget.frames_per_visit`: frames captured per VLM-selected visit.
- `exploration.visits_schedule`: per-round visit cap.
- `exploration.min_dist_schedule`: per-round spatial deduplication radius.

The default config uses relative paths so the project can be cloned anywhere.

## Running

Set your API key in the shell:

```bash
export ANTHROPIC_API_KEY="your_key_here"
```

Run the full GUI pipeline:

```bash
bash run_live.sh
```

Run baseline capture only:

```bash
bash run_live.sh --skip-vlm
```

Run without video export:

```bash
bash run_live.sh --skip-video
```

You can also run directly with Blender:

```bash
/Applications/Blender.app/Contents/MacOS/Blender \
  --python scripts/run_pipeline.py \
  -- \
  --config pipeline_config.json
```

## VLM Input Design

Each VLM call receives images plus lightweight textual labels:

- Observation frames: fixed FPS samples mixed with heuristic hint frames.
- Recent evidence frames: representative images from successful supplementary
  visits, labeled separately and not allowed as new targets.
- Previous visit history and blocked frame IDs to reduce repeated requests.

The model does not receive coverage ratios, coverage grids, priority scores,
room coordinates, or object coordinates. It chooses only a camera, a reference
frame ID, an approach, and a visual reason. The system then converts that visual
reference into concrete waypoints.

## Output

Each run creates:

```text
outputs/run_YYYYMMDD_HHMMSS/
├── captures/
├── analysis/
├── logs/run_summary.json
├── manifests/room_manifest.json
├── pipeline_config.json
└── scene_final.blend
```

These outputs are intentionally ignored by Git because they can be large and may
contain local absolute paths in generated logs.

## Notes

- Do not commit API keys. The launch scripts read keys from environment
  variables only.
- Do not commit `.vendor/`; it is a local dependency directory for Blender.
- Large GLB assets in `Meshy/` are ignored by default. Add small sample assets
  only if their license and size are suitable for the repository.

For questions, asset access, or reproduction details, contact
`zc569@cornell.edu`.
