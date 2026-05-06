"""
video_export.py
Phase 6: Converts PNG frame sequences to MP4 videos using ffmpeg.
Purely for visualization / review; not used as pipeline control input.
"""

import subprocess
import json
from pathlib import Path


class VideoExporter:
    def __init__(self, config: dict, output_dir: Path):
        self.video_cfg = config["video"]
        self.fps = self.video_cfg["fps"]
        self.codec = self.video_cfg["codec"]
        self.crf = self.video_cfg["crf"]
        self.videos_dir = output_dir / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self, capture_summaries: dict, round_id: str = "base"):
        """Export a video for each camera from its frame directory."""
        results = {}
        for cam_id, summary in capture_summaries.items():
            frames_dir = Path(summary["output_dir"])
            out_path = self.videos_dir / f"{round_id}_{cam_id}.mp4"
            ok = self._export(frames_dir, out_path)
            results[cam_id] = {
                "cam_id": cam_id,
                "round": round_id,
                "video_path": str(out_path),
                "success": ok
            }
        return results

    def export_overview(self, video_paths: list, out_name: str = "overview.mp4") -> bool:
        """Concatenate multiple videos into an overview clip."""
        out_path = self.videos_dir / out_name
        list_file = self.videos_dir / "_concat_list.txt"
        with open(list_file, "w") as f:
            for vp in video_paths:
                f.write(f"file '{vp}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out_path)
        ]
        ok = self._run(cmd, label="overview")
        list_file.unlink(missing_ok=True)
        return ok

    # ------------------------------------------------------------------ #
    def _export(self, frames_dir: Path, out_path: Path) -> bool:
        """Run ffmpeg on a directory of frame_XXXX.png files."""
        pattern = str(frames_dir / "frame_%04d.png")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(self.fps),
            "-i", pattern,
            "-vcodec", self.codec,
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p",
            str(out_path)
        ]
        return self._run(cmd, label=out_path.stem)

    @staticmethod
    def _run(cmd: list, label: str) -> bool:
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0:
                print(f"[VideoExporter] ✓ {label}")
                return True
            else:
                print(f"[VideoExporter] ✗ {label}: {result.stderr[-300:]}")
                return False
        except FileNotFoundError:
            print("[VideoExporter] ERROR: ffmpeg not found. Install ffmpeg and add to PATH.")
            return False
        except subprocess.TimeoutExpired:
            print(f"[VideoExporter] TIMEOUT: {label}")
            return False
