"""
trajectory_planner.py
Trajectory planner for three drone cameras.

Supported patterns:
  longitudinal_sweep  - sweeps along the Y axis (long axis)
  orbit               - elliptical orbit around room center
  perimeter           - travels along the inner wall boundary, always facing inward
  diagonal            - straight diagonal crossing
  grid                - lawnmower scan pattern

Improvements (2026-04):
  [TRAJ-1] diagonal: new look_mode parameter controls camera heading along path.
    - "forward"  : original — always face direction of travel (1 angle sector/cell)
    - "center"   : dynamically look toward room center (0,0) — 4-5 sectors/cell
    - "sweep"    : oscillate ±90° perpendicular to travel — 5+ sectors/cell (new default)
    Rationale: the original fixed heading means 80 frames contribute the same
    angle sector to every cell, making most frames redundant for coverage scoring.

  [TRAJ-2] orbit: new dual_radius option.
    When dual_radius=true, camera_03 flies two concentric orbits:
    inner orbit (first half of frames) covers room center,
    outer orbit (second half) covers wall-adjacent zones.
    Rationale: single-radius orbit only covers center band, missing corners.
"""

import math
from typing import List, Tuple

# (x, y, z, rx_rad, ry_rad, rz_rad)
Waypoint = Tuple[float, float, float, float, float, float]


class TrajectoryPlanner:
    def __init__(self, config: dict, room_manifest: dict):
        self.traj_cfg = config["trajectory"]
        self.cam_cfg  = config["cameras"]
        self.bounds   = room_manifest["room"]["bounds"]
        self.room     = room_manifest["room"]
        self.margin   = self.traj_cfg.get("boundary_margin", 0.5)
        self.h_range  = self.traj_cfg.get("height_range", [0.9, 3.2])

    # ------------------------------------------------------------------ #
    def plan_all(self) -> dict:
        """Generate baseline trajectories for all cameras."""
        xb = self.bounds["x"]
        yb = self.bounds["y"]
        zb = self.bounds["z"]
        room_w = round(xb[1] - xb[0], 3)
        room_d = round(yb[1] - yb[0], 3)
        room_h = round(zb[1] - zb[0], 3)
        diag   = round((room_w**2 + room_d**2) ** 0.5, 3)
        print(f"[TrajectoryPlanner] Room size: {room_w}m (W) x {room_d}m (D) x {room_h}m (H)")
        print(f"[TrajectoryPlanner] Floor diagonal: {diag}m  |  "
              f"X: [{xb[0]:.2f}, {xb[1]:.2f}]  "
              f"Y: [{yb[0]:.2f}, {yb[1]:.2f}]")
        print(f"[TrajectoryPlanner] Boundary margin: {self.margin}m  |  "
              f"Effective flight area: "
              f"{round(room_w - 2*self.margin, 2)}m x {round(room_d - 2*self.margin, 2)}m")

        plans = {}
        for drone_cfg in self.cam_cfg["drones"]:
            cam_id   = drone_cfg["id"]
            cam_traj = self.traj_cfg.get(cam_id)
            if cam_traj is None:
                print(f"[TrajectoryPlanner] WARNING: no trajectory config for {cam_id}, skipping.")
                continue
            pattern   = cam_traj["pattern"]
            waypoints = self._generate(pattern, cam_traj)
            plans[cam_id] = waypoints

            if pattern == "diagonal" and waypoints:
                x0, y0 = waypoints[0][0], waypoints[0][1]
                x1, y1 = waypoints[-1][0], waypoints[-1][1]
                direction = cam_traj.get("direction", "SW_to_NE")
                look_mode = cam_traj.get("look_mode", "sweep")
                print(f"[TrajectoryPlanner] {cam_id} ({drone_cfg['role']}): "
                      f"{pattern} [{direction}] look_mode={look_mode}  "
                      f"({x0:.2f},{y0:.2f}) -> ({x1:.2f},{y1:.2f})  "
                      f"height={cam_traj.get('height', 1.4)}m  "
                      f"frames={len(waypoints)}")
            else:
                print(f"[TrajectoryPlanner] {cam_id} ({drone_cfg['role']}): "
                      f"{pattern} -> {len(waypoints)} waypoints")
        return plans

    def plan_supplementary(self, cam_id: str, target: dict,
                           n_frames: int = 20) -> List[Waypoint]:
        """Generate a supplementary orbit trajectory around a VLM target area."""
        cx     = target.get("center_x", 0.0)
        cy     = target.get("center_y", 0.0)
        h      = target.get("height", 1.5)
        radius = target.get("radius", 0.8)

        waypoints = []
        for i in range(n_frames):
            angle = 2 * math.pi * i / n_frames
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            z = h
            x, y, z = self._clamp(x, y, z)
            rz = angle + math.pi   # always face the target center
            rx = math.radians(60)
            waypoints.append((x, y, z, rx, 0.0, rz))
        return waypoints

    # ------------------------------------------------------------------ #
    def _generate(self, pattern: str, cam_traj: dict) -> List[Waypoint]:
        dispatch = {
            "longitudinal_sweep": self._longitudinal_sweep,
            "orbit":              self._orbit,
            "perimeter":          self._perimeter,
            "diagonal":           self._diagonal,
            "grid":               self._grid,
        }
        fn = dispatch.get(pattern)
        if fn is None:
            raise ValueError(f"Unknown trajectory pattern: '{pattern}'")
        return fn(cam_traj)

    # ------------------------------------------------------------------ #
    # longitudinal_sweep
    # ------------------------------------------------------------------ #
    def _longitudinal_sweep(self, c: dict) -> List[Waypoint]:
        n       = c["total_frames"]
        x_off   = c["x_offset"]
        h       = c["height"]
        pitch_d = c["look_pitch_deg"]
        inward  = c.get("look_inward_offset", 0.0)
        reverse = c.get("reverse", False)

        yb = self.bounds["y"]
        y_start = yb[0] + self.margin
        y_end   = yb[1] - self.margin
        if reverse:
            y_start, y_end = y_end, y_start

        xb = self.bounds["x"]
        x  = max(xb[0] + self.margin, min(xb[1] - self.margin, x_off))
        z  = max(self.h_range[0], min(self.h_range[1], h))
        rx = math.radians(90.0 - pitch_d)

        waypoints = []
        for i in range(n):
            t = i / max(n - 1, 1)
            y = y_start + t * (y_end - y_start)

            move_dir = math.copysign(1.0, y_end - y_start)
            target_x = -x_off * 0.5 + inward
            target_y = y + move_dir * 0.5

            rz = math.atan2(target_y - y, target_x - x)
            waypoints.append((x, y, z, rx, 0.0, rz))

        return waypoints

    # ------------------------------------------------------------------ #
    # orbit
    # [TRAJ-2] dual_radius option: inner orbit (first half) + outer orbit (second half)
    # ------------------------------------------------------------------ #
    def _orbit(self, c: dict) -> List[Waypoint]:
        n            = c["total_frames"]
        h            = c["height"]
        radius_ratio = c.get("radius_ratio", 0.4)
        pitch_d      = c["look_pitch_deg"]
        dual_radius  = c.get("dual_radius", False)

        xb = self.bounds["x"]
        yb = self.bounds["y"]
        half_w = (xb[1] - xb[0]) / 2 - self.margin
        half_d = (yb[1] - yb[0]) / 2 - self.margin

        z  = max(self.h_range[0], min(self.h_range[1], h))
        rx = math.radians(90.0 - pitch_d)

        waypoints = []
        for i in range(n):
            angle = 2 * math.pi * i / n

            if dual_radius:
                # [TRAJ-2] First half: inner orbit (room center coverage)
                #          Second half: outer orbit (wall-zone coverage)
                if i < n // 2:
                    inner_ratio = radius_ratio * 0.55
                    r_x = half_w * inner_ratio * 2
                    r_y = half_d * inner_ratio * 2
                    # Use full angle for inner orbit
                    orbit_angle = 2 * math.pi * i / (n // 2)
                else:
                    outer_ratio = radius_ratio * 1.45
                    r_x = half_w * outer_ratio * 2
                    r_y = half_d * outer_ratio * 2
                    # Use full angle for outer orbit
                    orbit_angle = 2 * math.pi * (i - n // 2) / (n - n // 2)
                x = r_x * math.cos(orbit_angle)
                y = r_y * math.sin(orbit_angle)
                rz = orbit_angle + math.pi
            else:
                # Original single-radius orbit
                r_x = half_w * radius_ratio * 2
                r_y = half_d * radius_ratio * 2
                x = r_x * math.cos(angle)
                y = r_y * math.sin(angle)
                rz = angle + math.pi   # face toward center

            x, y, z_c = self._clamp(x, y, z)
            waypoints.append((x, y, z_c, rx, 0.0, rz))

        return waypoints

    # ------------------------------------------------------------------ #
    # perimeter
    # ------------------------------------------------------------------ #
    def _perimeter(self, c: dict) -> List[Waypoint]:
        n       = c.get("total_frames", 80)
        h       = c.get("height", 1.4)
        pitch_d = c.get("look_pitch_deg", 65)

        xb = self.bounds["x"]
        yb = self.bounds["y"]
        m  = self.margin
        z  = max(self.h_range[0], min(self.h_range[1], h))
        rx = math.radians(90.0 - pitch_d)

        corners = [
            (xb[0]+m, yb[0]+m),
            (xb[1]-m, yb[0]+m),
            (xb[1]-m, yb[1]-m),
            (xb[0]+m, yb[1]-m),
        ]
        pts = self._interpolate_polygon(corners, n)

        waypoints = []
        for x, y in pts:
            rz = math.atan2(-y, -x)   # face origin (room center)
            waypoints.append((x, y, z, rx, 0.0, rz))
        return waypoints

    # ------------------------------------------------------------------ #
    # diagonal
    #
    # [TRAJ-1] look_mode parameter:
    #
    #   "forward" (original):
    #     Camera always faces direction of travel. rz is fixed for the
    #     entire trajectory. Every frame contributes the same angle sector
    #     to every cell → 1 sector per cell regardless of frame count.
    #
    #   "center" (new):
    #     Camera dynamically looks toward room center (0,0) from each
    #     position. rz rotates from ~NE at start to ~SW at end (or vice
    #     versa). Covers 4-5 distinct angle sectors per cell.
    #
    #   "sweep" (new default):
    #     Camera oscillates ±90° perpendicular to travel direction,
    #     completing n_sweeps full oscillations over the path.
    #     Covers 5+ angle sectors per cell in a predictable, uniform pattern.
    #     Combined with camera_02's crossing diagonal, all 8 sectors get
    #     covered across the room floor.
    # ------------------------------------------------------------------ #
    def _diagonal(self, c: dict) -> List[Waypoint]:
        n         = c.get("total_frames", 80)
        h_start   = c.get("height", 1.4)
        h_end     = c.get("height_end", h_start)
        pitch_d   = c.get("look_pitch_deg", 65)
        direction = c.get("direction", "SW_to_NE")
        look_mode = c.get("look_mode", "sweep")   # [TRAJ-1] new parameter
        n_sweeps  = c.get("n_sweeps", 3)           # sweep oscillations (sweep mode only)

        xb = self.bounds["x"]
        yb = self.bounds["y"]
        m  = self.margin

        SW = (xb[0]+m, yb[0]+m)
        NE = (xb[1]-m, yb[1]-m)
        SE = (xb[1]-m, yb[0]+m)
        NW = (xb[0]+m, yb[1]-m)

        corners = {
            "SW_to_NE": (SW, NE),
            "NE_to_SW": (NE, SW),
            "SE_to_NW": (SE, NW),
            "NW_to_SE": (NW, SE),
        }

        if direction not in corners:
            raise ValueError(
                f"Unknown diagonal direction: '{direction}'. "
                f"Choose from: {list(corners.keys())}"
            )

        (x0, y0), (x1, y1) = corners[direction]

        # Travel direction angle — used as fallback and sweep base
        travel_rz = math.atan2(y1 - y0, x1 - x0)
        rx        = math.radians(90.0 - pitch_d)

        waypoints = []
        for i in range(n):
            t = i / max(n - 1, 1)

            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            h = h_start + t * (h_end - h_start)
            z = max(self.h_range[0], min(self.h_range[1], h))

            # ── [TRAJ-1] Heading calculation by look_mode ─────────────
            if look_mode == "center":
                # Look toward room center (0, 0).
                # Falls back to travel direction when too close to center
                # to avoid atan2(0, 0) instability.
                dist_to_center = math.hypot(x, y)
                if dist_to_center > 0.4:
                    rz = math.atan2(-y, -x)
                else:
                    rz = travel_rz

            elif look_mode == "sweep":
                # Oscillate ±90° relative to travel direction.
                # At t=0: facing left of travel (travel_rz + 90°)
                # At t=0.5/n_sweeps: facing travel direction
                # At t=1/n_sweeps: facing right of travel (travel_rz - 90°)
                # Repeats n_sweeps times over the full diagonal.
                sweep = math.sin(2 * math.pi * t * n_sweeps) * (math.pi / 2)
                rz = travel_rz + sweep

            else:
                # "forward": original fixed heading
                rz = travel_rz
            # ─────────────────────────────────────────────────────────

            waypoints.append((x, y, z, rx, 0.0, rz))

        return waypoints

    # ------------------------------------------------------------------ #
    # grid - lawnmower scan pattern
    # ------------------------------------------------------------------ #
    def _grid(self, c: dict) -> List[Waypoint]:
        n       = c.get("total_frames", 80)
        h       = c.get("height", 2.0)
        pitch_d = c.get("look_pitch_deg", 80)
        rows    = c.get("rows", 5)

        xb = self.bounds["x"]
        yb = self.bounds["y"]
        m  = self.margin
        z  = max(self.h_range[0], min(self.h_range[1], h))
        rx = math.radians(90.0 - pitch_d)

        pts_per_row = max(n // rows, 2)
        xs = [xb[0]+m + (xb[1]-xb[0]-2*m)*j/(pts_per_row-1)
              for j in range(pts_per_row)]
        ys = [yb[0]+m + (yb[1]-yb[0]-2*m)*i/(rows-1) for i in range(rows)]

        waypoints = []
        for i, y in enumerate(ys):
            row_xs = xs if i % 2 == 0 else list(reversed(xs))
            for x in row_xs:
                waypoints.append((x, y, z, rx, 0.0, 0.0))

        return waypoints[:n]

    # ------------------------------------------------------------------ #
    def _clamp(self, x, y, z):
        xb, yb = self.bounds["x"], self.bounds["y"]
        m = self.margin
        x = max(xb[0]+m, min(xb[1]-m, x))
        y = max(yb[0]+m, min(yb[1]-m, y))
        z = max(self.h_range[0], min(self.h_range[1], z))
        return x, y, z

    @staticmethod
    def _interpolate_polygon(corners: list, n: int) -> list:
        """Uniformly sample n points along a closed polygon path."""
        segs  = []
        total = 0.0
        for i in range(len(corners)):
            a = corners[i]
            b = corners[(i+1) % len(corners)]
            d = math.hypot(b[0]-a[0], b[1]-a[1])
            segs.append((a, b, d, total))
            total += d
        points = []
        for i in range(n):
            t_target = total * i / n
            for (a, b, d, cum) in segs:
                if cum+d >= t_target or d == 0:
                    local_t = (t_target-cum)/d if d > 0 else 0
                    px = a[0] + local_t*(b[0]-a[0])
                    py = a[1] + local_t*(b[1]-a[1])
                    points.append((px, py))
                    break
        return points
