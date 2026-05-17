"""
vlm_controller.py
Phase 8: Pure visual decision making.

The VLM receives ONLY keyframe images. It has no access to:
- coverage ratios or grid data
- coordinate values
- priority scores

Its job is purely visual: look at the images and identify what looks
geometrically incomplete, dark, or missing. The system handles all
geometry and termination decisions independently.

Spatial referencing: VLM references a specific keyframe by ID.
The system reads that frame's pose_log to compute where to fly next.

Fix log (2026-04):
  [FIX-2] decide() now accepts blocked_frames parameter — a list of
          reference_frame IDs that were rejected by spatial dedup.
          These are injected into the prompt so VLM stops re-proposing
          frames that will never be executed.
  [FIX-3] camera_03 description updated to convey its area-coverage
          efficiency advantage, steering VLM toward it for large gaps.
"""

import json
import base64
import os
import re
from pathlib import Path
from typing import Optional


class VLMController:
    def __init__(self, config: dict, output_dir: Path):
        self.cfg          = config["vlm"]
        self.budget_cfg   = config.get("budget", {})
        self.analysis_dir = output_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def decide(self, keyframes: list, budget_summary: dict,
               visit_history: list = None,
               blocked_frames: list = None) -> dict:
        """
        Call VLM with keyframe images only.
        Returns a list of visual targets for supplementary capture.

        Args:
            keyframes:      list of {camera, frame_id, image_path}
            budget_summary: current budget state
            visit_history:  executed visits [{round, camera, approach,
                            reference_frame, problem}, ...]
            blocked_frames: [FIX-2] reference_frame IDs rejected by spatial
                            dedup — VLM must not re-propose these.
        """
        images  = self._load_images(keyframes)
        history = visit_history  or []
        blocked = blocked_frames or []

        # Save readable input log (no image data)
        with open(self.analysis_dir / "vlm_input.json", "w") as f:
            json.dump({
                "keyframe_count":  len(keyframes),
                "keyframes_sent":  [
                    {k: v for k, v in kf.items() if k != "b64_jpeg"}
                    for kf in images
                ],
                "budget":          budget_summary,
                "visit_history":   history,
                "blocked_frames":  blocked,          # [FIX-2] logged for debugging
            }, f, indent=2)

        print(f"[VLMController] Sending {len(images)} keyframes to "
              f"{self.cfg.get('model', 'unknown')} "
              f"({self.cfg.get('provider', 'anthropic')})  "
              f"blocked_frames={len(blocked)} ...")
        decision = self._call_vlm(images, budget_summary, history, blocked)

        with open(self.analysis_dir / "vlm_decision.json", "w") as f:
            json.dump(decision, f, indent=2)
        # Print reasoning so it's visible in the run log for debugging
        if decision.get("reasoning"):
            print(f"[VLMController] Reasoning: {decision['reasoning'][:200]}...")
        print(f"[VLMController] Decision: {decision.get('overall_assessment', '')}")
        return decision

    # ------------------------------------------------------------------ #
    #  Image loading
    # ------------------------------------------------------------------ #
    def _load_images(self, keyframes: list) -> list:
        images = []
        for kf in keyframes:
            img_path = Path(kf.get("image_path", ""))
            if not img_path.exists():
                continue
            b64 = self._encode_image(img_path)
            if b64:
                images.append({**kf, "b64_jpeg": b64})
        return images

    # ------------------------------------------------------------------ #
    #  VLM call dispatch
    # ------------------------------------------------------------------ #
    def _call_vlm(self, images: list, budget_summary: dict,
                  visit_history: list = None,
                  blocked_frames: list = None) -> dict:
        provider = self.cfg.get("provider", "anthropic")
        history  = visit_history  or []
        blocked  = blocked_frames or []
        if provider == "anthropic":
            return self._call_anthropic(images, budget_summary, history, blocked)
        if provider == "qwen":
            return self._call_qwen(images, budget_summary, history, blocked)
        raise ValueError(
            f"Unsupported VLM provider: '{provider}'. "
            f"Choose 'anthropic' or 'qwen'."
        )

    def _call_anthropic(self, images: list, budget_summary: dict,
                        visit_history: list = None,
                        blocked_frames: list = None) -> dict:
        try:
            import anthropic
        except ImportError:
            print("[VLMController] anthropic not installed. Using mock decision.")
            return self._mock_decision(images)

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("[VLMController] ANTHROPIC_API_KEY not set. Using mock decision.")
            return self._mock_decision(images)

        client = anthropic.Anthropic(api_key=api_key)

        content = []
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": "image/jpeg",
                    "data":       img["b64_jpeg"]
                }
            })
            content.append({
                "type": "text",
                "text": f"[{img['frame_id']} | {img['camera']}]"
            })

        content.append({
            "type": "text",
            "text": self._build_prompt(
                images, budget_summary,
                visit_history or [],
                blocked_frames or [],
            )
        })

        response = client.messages.create(
            model=self.cfg.get("model", "claude-opus-4-5"),
            max_tokens=self.cfg.get("max_tokens", 2048),
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}]
        )

        return self._parse_response(response.content[0].text)

    def _call_qwen(self, images: list, budget_summary: dict,
                   visit_history: list = None,
                   blocked_frames: list = None) -> dict:
        """
        Call Qwen VL API via the OpenAI-compatible DashScope endpoint.

        Required env var: DASHSCOPE_API_KEY
        Recommended model: qwen-vl-max  (or qwen-vl-plus for lower cost)
        """
        try:
            from openai import OpenAI
        except ImportError:
            print("[VLMController] openai package not installed. "
                  "Run: pip install openai")
            return self._mock_decision(images)

        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            print("[VLMController] DASHSCOPE_API_KEY not set. Using mock decision.")
            return self._mock_decision(images)

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )

        user_content = []
        for img in images:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img['b64_jpeg']}"
                }
            })
            user_content.append({
                "type": "text",
                "text": f"[{img['frame_id']} | {img['camera']}]"
            })

        user_content.append({
            "type": "text",
            "text": self._build_prompt(
                images, budget_summary,
                visit_history or [],
                blocked_frames or [],
            )
        })

        response = client.chat.completions.create(
            model=self.cfg.get("model", "qwen-vl-max"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content}
            ],
            max_tokens=self.cfg.get("max_tokens", 2048)
        )

        raw_text = response.choices[0].message.content
        return self._parse_response(raw_text)

    # ------------------------------------------------------------------ #
    #  Prompt builder
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_prompt(images: list, budget_summary: dict,
                      visit_history: list = None,
                      blocked_frames: list = None) -> str:
        frames_remaining = budget_summary.get("frames_remaining", "?")
        visits_remaining = budget_summary.get("visits_remaining", "?")
        cameras_allowed  = budget_summary.get("cameras_allowed", [])
        frames_per_visit = budget_summary.get("frames_per_visit", 20)
        # Per-round cap: how many visits will actually be executed this round.
        # This is always <= visits_remaining (global leftover).
        visits_this_round = budget_summary.get("visits_per_round", visits_remaining)
        history  = visit_history  or []
        blocked  = blocked_frames or []

        # camera_03 description emphasises area-coverage efficiency
        cam_lines = []
        for cam in cameras_allowed:
            desc = {
                "camera_01": (
                    "low altitude (~1.4m) — best for close-up object detail "
                    "and side views of a specific item"
                ),
                "camera_02": (
                    "low altitude (~1.6m) — best for close-up object detail "
                    "and side views of a specific item"
                ),
                "camera_03": (
                    "high altitude (~2.9m) — MOST EFFICIENT for covering large "
                    "floor areas and corners in a single visit; use this "
                    "whenever a whole region (not a single object) looks "
                    "missing or absent from all keyframes"
                ),
            }.get(cam, cam)
            cam_lines.append(f"  - {cam}: {desc}")
        cam_desc = "\n".join(cam_lines) if cam_lines else "  (none available)"

        frame_refs = "\n".join(
            f"  {img['frame_id']} ({img['camera']})"
            for img in images
        )

        # Executed visit history
        if history:
            history_lines = [
                f"  Visit {i+1}: {v['camera']} {v['approach']} "
                f"referenced {v['reference_frame']} — \"{v['problem']}\""
                for i, v in enumerate(history)
            ]
            history_section = "\n".join(history_lines)
        else:
            history_section = "  (none — this is the first round)"

        # Blocked frames — rejected by spatial deduplication
        if blocked:
            blocked_section = "\n".join(f"  - {f}" for f in blocked)
        else:
            blocked_section = "  (none)"

        return f"""
## Already Captured (Previous Visits)
The following locations have already been visited. Do NOT select the same
reference_frame or a nearby area — those are already covered.

{history_section}

## Blocked Frames — DO NOT Re-Select
These reference_frames were proposed before but REJECTED by the system because
they are too close to already-visited locations. Selecting them again will have
no effect. Choose a completely different frame showing a different area.

{blocked_section}

## Your Budget This Round
You may submit AT MOST {visits_this_round} visit(s) this round.
Each visit captures {frames_per_visit} frames. Total frames remaining: {frames_remaining}.

Cameras available:
{cam_desc}

## Keyframe Reference List
{frame_refs}

## Your Task

Step 1 — Survey the images and list every area that still looks visually
incomplete (dark, flat, blurry, or simply absent from all keyframes).
Exclude anything already addressed by previous visits or blocked frames.

Step 2 — From that list, select the {visits_this_round} MOST IMPORTANT
gaps only. Quality over quantity: a small focused list is better than a
long list where only the first few will be used.

Step 3 — Order your visits from most important to least important.
The system executes them in order and may stop early, so put your
highest-value visit first.

Rules:
- Maximum {visits_this_round} visits in your response.
- Each visit must target a VISUALLY DISTINCT area. If two keyframes
  show the same scene content, pick only one — do not propose visits
  for both.
- Do NOT re-use any frame_id from the blocked list.
- For large uncovered regions: use camera_03 orbit_high.
- For specific objects seen from only one side: use camera_01 or
  camera_02 orbit_close.

If no meaningful gaps remain, set should_capture to false.

Output ONLY this JSON (no markdown, no explanation).
The "reasoning" field must be filled first — write your visual analysis
there before deciding on visits. This ensures your visit choices follow
from what you actually observed in the images.

{{
  "reasoning": "Go through each keyframe one by one. Note what looks dark, flat, or absent. Then identify which gaps are NOT already covered by previous visits or blocked frames. Finally, rank the remaining gaps by importance and explain why.",
  "should_capture": true or false,
  "overall_assessment": "one sentence describing the most critical remaining gap",
  "visits": [
    {{
      "camera":          "camera_01" or "camera_02" or "camera_03",
      "reference_frame": "a frame_id NOT in the blocked list",
      "approach":        "orbit_close" or "orbit_high" or "orbit_wide",
      "problem":         "specific description of what is missing here"
    }}
  ],
  "stop_reason": null or "scene_looks_complete" or "budget_insufficient"
}}
"""

    # ------------------------------------------------------------------ #
    #  Response parsing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_response(raw_text: str) -> dict:
        text  = raw_text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            print(f"[VLMController] Could not parse response:\n{raw_text[:400]}")
            return {
                "should_capture":     False,
                "overall_assessment": "Parse error — could not read VLM response",
                "visits":             [],
                "stop_reason":        "parse_error",
                "raw_response":       raw_text[:1000]
            }

    # ------------------------------------------------------------------ #
    #  Mock fallback
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mock_decision(images: list) -> dict:
        if not images:
            return {
                "should_capture":     False,
                "overall_assessment": "No images available (mock)",
                "visits":             [],
                "stop_reason":        "no_images"
            }
        return {
            "should_capture":     True,
            "overall_assessment": "Mock decision — requesting one close-orbit visit",
            "visits": [{
                "camera":          "camera_01",
                "reference_frame": images[0]["frame_id"],
                "approach":        "orbit_close",
                "problem":         "Mock: supplementary pass of first keyframe area"
            }],
            "stop_reason": None
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _encode_image(path: Path, max_dim: int = 512) -> Optional[str]:
        """Load image and return base64 JPEG. Uses PIL if available."""
        try:
            from PIL import Image
            import io
            img = Image.open(path).convert("RGB")
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except ImportError:
            pass
        except Exception as e:
            print(f"[VLMController] PIL error on {path}: {e}")

        try:
            with open(path, "rb") as f:
                raw = f.read()
            return base64.b64encode(raw).decode("utf-8")
        except Exception as e:
            print(f"[VLMController] Could not encode {path}: {e}")
            return None


# ------------------------------------------------------------------ #
#  System prompt
# ------------------------------------------------------------------ #
SYSTEM_PROMPT = """
You are the visual inspector for an indoor 3D reconstruction drone system.
Your only job is to look at images and identify what is visually incomplete.

You do NOT have access to:
- Coverage maps or statistics
- Room coordinates or measurements
- Any quantitative analysis

You work purely from what you can SEE. Trust your visual judgment:
if something looks incomplete, flag it. If everything looks well-captured,
say so even if you have budget remaining.

Important rules:
- Always fill the "reasoning" field first. Analyze each keyframe before
  deciding on visits — do not skip straight to the visit list.
- If a reference_frame appears in the "Blocked Frames" list in the user
  prompt, do NOT select it. It will be rejected regardless.
- Always pick a frame showing a genuinely different part of the scene.

Return ONLY valid JSON. No markdown, no prose outside the JSON.
""".strip()