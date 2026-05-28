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

Reasoning probe (2026-05):
  After the VLM returns its JSON decision, a second, separate call asks
  the same VLM to *describe the algorithm or strategy it just used* to
  reach that decision. This is a "self-explanation" probe — useful as a
  starting hypothesis for understanding VLM behaviour, with the caveat
  that LLM self-explanations are known to be post-hoc rationalisations
  (Turpin et al. 2023). The answer is logged to vlm_self_reasoning.json
  for offline analysis and never feeds back into the pipeline.
  Toggle via cfg["vlm"]["probe_reasoning"] (default: True).
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
               blocked_frames: list = None,
               recent_evidence: list = None) -> dict:
        """
        Call VLM with observation frames plus optional recent evidence frames.
        Returns a list of visual targets for supplementary capture.

        Args:
            keyframes:       ordinary observation frames. This list may include
                             algorithm-selected hint frames, but that role is
                             intentionally hidden from the VLM so it does not
                             overfit to the algorithm's suggestions.
            budget_summary:  current budget state
            visit_history:   executed visits [{round, camera, approach,
                             reference_frame, problem}, ...]
            blocked_frames:  [FIX-2] reference_frame IDs rejected by spatial
                             dedup — VLM must not re-propose these.
            recent_evidence: representative frames from successful recent
                             supplementary visits. These are labeled separately
                             so the VLM can visually confirm what has already
                             been covered and avoid repeated requests.
        """
        observation_images = self._load_images(keyframes)
        recent_images = self._load_images(recent_evidence or [])
        images = observation_images + recent_images
        history = visit_history  or []
        blocked = blocked_frames or []

        # Save readable input log (no image data)
        with open(self.analysis_dir / "vlm_input.json", "w") as f:
            json.dump({
                "observation_frame_count": len(keyframes),
                "recent_evidence_count":   len(recent_evidence or []),
                "frames_sent":  [
                    {k: v for k, v in kf.items() if k != "b64_jpeg"}
                    for kf in images
                ],
                "budget":          budget_summary,
                "visit_history":   history,
                "blocked_frames":  blocked,          # [FIX-2] logged for debugging
            }, f, indent=2)

        print(f"[VLMController] Sending {len(images)} frames to "
              f"{self.cfg.get('model', 'unknown')} "
              f"({self.cfg.get('provider', 'anthropic')})  "
              f"recent_evidence={len(recent_images)}  "
              f"blocked_frames={len(blocked)} ...")
        decision = self._call_vlm(images, budget_summary, history, blocked)

        with open(self.analysis_dir / "vlm_decision.json", "w") as f:
            json.dump(decision, f, indent=2)
        # Print reasoning so it's visible in the run log for debugging
        if decision.get("reasoning"):
            print(f"[VLMController] Reasoning: {decision['reasoning'][:200]}...")
        print(f"[VLMController] Decision: {decision.get('overall_assessment', '')}")

        # ---- Reasoning probe ----------------------------------------- #
        # Ask the SAME VLM to describe the algorithm it just used.
        # This is logged for offline analysis (revealed-preferences study).
        # It does NOT feed back into the pipeline.
        if self.cfg.get("probe_reasoning", True):
            try:
                self._probe_reasoning(images, budget_summary, history,
                                      blocked, decision)
            except Exception as e:
                print(f"[VLMController] Reasoning probe failed: {e}")

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
            encoded = self._encode_image(img_path)
            if not encoded:
                continue
            b64, media_type = encoded
            if b64:
                images.append({**kf, "b64_jpeg": b64, "media_type": media_type})
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
                    "media_type": img.get("media_type", "image/jpeg"),
                    "data":       img["b64_jpeg"]
                }
            })
            content.append({
                "type": "text",
                "text": self._frame_label(img)
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
                    "url": (f"data:{img.get('media_type', 'image/jpeg')};"
                            f"base64,{img['b64_jpeg']}")
                }
            })
            user_content.append({
                "type": "text",
                "text": self._frame_label(img)
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
    #  Frame labeling
    # ------------------------------------------------------------------ #
    @staticmethod
    def _frame_label(img: dict) -> str:
        """Keep recent evidence visually distinct from targetable frames."""
        if img.get("_recent_evidence"):
            return f"[RECENT_EVIDENCE | {img['frame_id']} | {img['camera']}]"
        return f"[{img['frame_id']} | {img['camera']}]"

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
        round_index = budget_summary.get("round_index", "?")
        max_rounds = budget_summary.get("max_rounds", "?")
        # Per-round cap: how many visits will actually be executed this round.
        # This is always <= visits_remaining (global leftover).
        visits_this_round = budget_summary.get("visits_per_round", visits_remaining)
        history  = visit_history  or []
        blocked  = blocked_frames or []

        if round_index == 1:
            round_focus = """
## Round Focus
This is Round 1: prioritize broad room coverage and visually complete framing.
Select large missing room areas, corners, walls, floor bands, and regions absent
from all keyframes before selecting close object detail. Prefer camera_03
orbit_high for broad coverage. Pick object close-ups only if an object is barely
visible or strongly occluded.
""".strip()
        elif round_index == 2:
            round_focus = """
## Round Focus
This is Round 2: balance remaining room coverage with important object detail.
Include at least one broad coverage visit if any corner, wall-side region, or
large empty-looking area still appears incomplete. Then use close orbits for
objects that clearly lack side/back views. Do not spend the whole round on the
same object.
""".strip()
        elif round_index == 3:
            round_focus = """
## Round Focus
This is Round 3: improve final visual quality with the few highest-impact gaps.
Choose visits that make the final scene look more complete: obvious empty
corners, poorly separated objects, missing side/back views, or strong
occlusions. Do not repeat close-ups of objects already covered unless no broad
coverage gaps remain.
""".strip()
        else:
            round_focus = """
## Round Focus
Prioritize broad coverage first, then object detail, and avoid repeating the
same object or location when other meaningful gaps are visible.
""".strip()

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
            if not img.get("_recent_evidence")
        )
        recent_refs = "\n".join(
            f"  {img['frame_id']} ({img['camera']})"
            for img in images
            if img.get("_recent_evidence")
        ) or "  (none yet — this is the first round or no recent evidence was available)"

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

        # Keep the model's written reasoning short. Earlier prompts asked for
        # frame-by-frame analysis, which helped debugging but made Claude spend
        # many output tokens and occasionally truncate the required JSON.
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
You are in round {round_index} of {max_rounds}.
You may submit AT MOST {visits_this_round} visit(s) this round.
Each visit captures {frames_per_visit} frames. Total frames remaining: {frames_remaining}.

Cameras available:
{cam_desc}

{round_focus}

## Observation Frames
These are the current visual context frames. Use ONLY these frame IDs as
reference_frame values in your output.

{frame_refs}

## Recent Evidence Frames
These frames show areas already captured by recent supplementary visits.
Use them as visual memory to avoid requesting the same area again. They are
NOT normal target candidates: do not use RECENT_EVIDENCE frame IDs as
reference_frame values.

{recent_refs}

## Your Task

Step 1 — Privately survey the images and identify areas that still look
visually incomplete (dark, overexposed, washed out, flat, blurry, occluded,
visible from only one side, or simply absent from all keyframes).
Exclude anything already addressed by previous visits or blocked frames.
First check broad room/corner/edge coverage, then check object details.
Do not write frame-by-frame notes in the JSON output.

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
- The reference_frame in every visit must come from Observation Frames, not
  Recent Evidence Frames.
- Broad room/corner/edge coverage outranks repeated close-ups of the same
  object. After an object has already been visited, prefer a different object
  or a broad room gap unless that object is still the only serious issue.
- Good lighting alone is NOT evidence that coverage is complete. If a region
  is bright but lacks shadows, depth cues, side/back views, or object
  separation, treat it as visually uncertain and consider a supplementary
  capture.
- Do not stop just because images look evenly lit. Stop only when the room
  areas and object surfaces appear geometrically complete from the visible
  viewpoints.
- For large uncovered regions: use camera_03 orbit_high.
- For specific objects seen from only one side: use camera_01 or
  camera_02 orbit_close.

If no meaningful gaps remain, set should_capture to false.

Output ONLY this JSON (no markdown, no explanation).
The "reasoning" field must be a concise visual summary, not a full
chain-of-thought. Limit it to 80 words or fewer. Mention only the top
2-3 remaining gaps and whether recent evidence or blocked frames changed
the decision.

{{
  "reasoning": "Concise summary under 80 words: top remaining broad coverage gaps first, then object detail gaps; mention any repeated/blocked areas avoided.",
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
    #  Reasoning probe — "how did you decide?"
    # ------------------------------------------------------------------ #
    #
    # Run AFTER the decision call. Feeds the same images + the model's own
    # JSON decision back to the same VLM, and asks an open-ended question
    # about the algorithm / strategy it used. The answer is appended to
    # vlm_self_reasoning.json (one entry per call).
    #
    # IMPORTANT CAVEAT: LLM self-explanations are well known to be
    # post-hoc rationalisations and are sycophantic (Turpin et al. 2023;
    # Sharma et al. 2023). This probe is therefore a hypothesis generator,
    # not ground truth. Behavioural validation should be done separately.
    # ------------------------------------------------------------------ #
    def _probe_reasoning(self, images: list, budget_summary: dict,
                         visit_history: list, blocked_frames: list,
                         decision: dict) -> None:
        provider = self.cfg.get("provider", "anthropic")
        probe_question = self._build_probe_prompt(
            images, budget_summary, visit_history, blocked_frames, decision
        )

        if provider == "anthropic":
            answer = self._probe_anthropic(images, probe_question)
        elif provider == "qwen":
            answer = self._probe_qwen(images, probe_question)
        else:
            answer = f"[unsupported provider: {provider}]"

        # Append (not overwrite) so multi-round runs keep full history
        log_path = self.analysis_dir / "vlm_self_reasoning.json"
        entries = []
        if log_path.exists():
            try:
                with open(log_path, "r") as f:
                    entries = json.load(f)
                if not isinstance(entries, list):
                    entries = [entries]
            except (json.JSONDecodeError, OSError):
                entries = []

        entries.append({
            "round_index":     budget_summary.get("round_index"),
            "frames_sent":     len(images),
            "decision_summary": decision.get("overall_assessment", ""),
            "decision_visits": [
                {
                    "camera":          v.get("camera"),
                    "approach":        v.get("approach"),
                    "reference_frame": v.get("reference_frame"),
                    "problem":         v.get("problem"),
                }
                for v in decision.get("visits", [])
            ],
            "self_reasoning":  answer,
        })

        with open(log_path, "w") as f:
            json.dump(entries, f, indent=2)

        print(f"[VLMController] Self-reasoning probed ({len(answer)} chars) "
              f"-> {log_path.name}")

    @staticmethod
    def _build_probe_prompt(images: list, budget_summary: dict,
                            visit_history: list, blocked_frames: list,
                            decision: dict) -> str:
        """Open-ended question about the algorithm / strategy used."""
        decision_str = json.dumps(
            {
                "should_capture":     decision.get("should_capture"),
                "overall_assessment": decision.get("overall_assessment"),
                "visits":             decision.get("visits", []),
                "stop_reason":        decision.get("stop_reason"),
            },
            indent=2,
        )
        return f"""
You just produced the following decision in response to the images and
context I showed you a moment ago:

{decision_str}

Now, in plain language, please answer ALL of the following questions
about how you arrived at that decision. Be as concrete and specific as
you can. Speculation about your own internals is welcome — say "I don't
know" if you truly don't.

1. What algorithm or decision strategy did you use? If your strategy has
   a name (e.g. frontier exploration, next-best-view, information-gain
   maximisation, greedy coverage, heuristic ranking, etc.), name it. If
   it does not match a known algorithm, describe the procedure step by
   step in your own words.

2. What objective (utility function / score) were you maximising, if
   any? Express it as concretely as possible — even a rough formula or
   ordered list of priorities helps.

3. What constraints did you treat as hard limits? What did you trade off
   against what?

4. Which specific frames or visual cues drove the decision, and why
   those rather than the alternatives?

5. If you ran this same decision 10 times, would you give the same
   answer? If not, what would vary, and what would stay fixed?

6. Honest self-assessment: are you reasonably confident your description
   above reflects what actually drove your choice, or is it possible
   you're reconstructing a plausible-sounding story after the fact?

Reply in plain prose. Use numbered sections matching the questions
above. Do NOT output JSON.
""".strip()

    def _probe_anthropic(self, images: list, probe_question: str) -> str:
        try:
            import anthropic
        except ImportError:
            return "[anthropic SDK not installed]"

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return "[ANTHROPIC_API_KEY not set]"

        client = anthropic.Anthropic(api_key=api_key)

        content = []
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": img.get("media_type", "image/jpeg"),
                    "data":       img["b64_jpeg"]
                }
            })
            content.append({
                "type": "text",
                "text": self._frame_label(img)
            })
        content.append({"type": "text", "text": probe_question})

        try:
            response = client.messages.create(
                model=self.cfg.get("model", "claude-sonnet-4-6"),
                max_tokens=self.cfg.get("probe_max_tokens", 2048),
                system=PROBE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"[anthropic probe error: {e}]"

    def _probe_qwen(self, images: list, probe_question: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            return "[openai SDK not installed]"

        api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            return "[DASHSCOPE_API_KEY not set]"

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )

        user_content = []
        for img in images:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": (f"data:{img.get('media_type', 'image/jpeg')};"
                            f"base64,{img['b64_jpeg']}")
                }
            })
            user_content.append({
                "type": "text",
                "text": self._frame_label(img)
            })
        user_content.append({"type": "text", "text": probe_question})

        try:
            response = client.chat.completions.create(
                model=self.cfg.get("model", "qwen-vl-max"),
                messages=[
                    {"role": "system", "content": PROBE_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_content},
                ],
                max_tokens=self.cfg.get("probe_max_tokens", 2048),
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[qwen probe error: {e}]"

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
    def _encode_image(path: Path, max_dim: int = 512) -> Optional[tuple]:
        """Return (base64, media_type), preserving correct MIME metadata."""
        try:
            from PIL import Image
            import io
            img = Image.open(path).convert("RGB")
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"
        except ImportError:
            pass
        except Exception as e:
            print(f"[VLMController] PIL error on {path}: {e}")

        try:
            with open(path, "rb") as f:
                raw = f.read()
            media_type = {
                ".png":  "image/png",
                ".jpg":  "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif":  "image/gif",
            }.get(path.suffix.lower(), "application/octet-stream")
            return base64.b64encode(raw).decode("utf-8"), media_type
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
- Before deciding, inspect the images carefully, but keep the written
  "reasoning" field concise. Do not output frame-by-frame analysis.
- If a reference_frame appears in the "Blocked Frames" list in the user
  prompt, do NOT select it. It will be rejected regardless.
- Always pick a frame showing a genuinely different part of the scene.

Return ONLY valid JSON. No markdown, no prose outside the JSON.
""".strip()


# ------------------------------------------------------------------ #
#  System prompt for the reasoning probe (self-explanation pass)
# ------------------------------------------------------------------ #
PROBE_SYSTEM_PROMPT = """
You are a vision-language model whose decision-making behaviour is being
studied. You have just produced a JSON decision for an indoor 3D
reconstruction next-best-view task. The researcher is now asking you to
explain — in plain language — what algorithm, objective, and constraints
you used to arrive at that decision.

Guidelines for your reply:
- Be specific and concrete. Name algorithms or strategies when you can.
- Distinguish between (a) what you actually used and (b) what you might
  expect a competent agent to use. If they differ, say so.
- If you are confabulating a plausible-sounding story rather than
  reporting an actual procedure, please say "I am not sure this reflects
  my real process" and explain what you do know.
- It is acceptable — even useful — to say "I don't know" for parts you
  cannot introspect.
- Reply in plain prose with numbered sections matching the questions.
  Do NOT output JSON. Do NOT change or restate the original decision.
""".strip()
