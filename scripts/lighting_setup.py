"""
lighting_setup.py
Phase 2: Automatic indoor lighting initialization.
"""

import bpy
import math
import random
from typing import List


class LightingSetup:
    def __init__(self, config: dict, room_manifest: dict):
        self.cfg  = config["lighting"]
        self.room = room_manifest["room"]   # store full room dict

    def setup(self):
        """Add main overhead light + optional aux fill lights."""
        self._remove_existing_lights()

        # Support both manifest formats: size list vs width/depth/height fields
        room = self.room
        if "size" in room:
            lx, ly, lz = room["size"]
        else:
            lx = room["width"]
            ly = room["depth"]
            lz = room["height"]

        # ---- Main area light (overhead) ----
        main_height = min(self.cfg["main_light_height"], lz - 0.1)
        self._add_light(
            name="MainLight",
            light_type="AREA",
            location=(0, 0, main_height),
            energy=self.cfg["main_light_energy"],
            size=max(lx, ly) * 0.6,
            color=(1.0, 0.97, 0.90)          # warm white
        )

        # ---- Auxiliary point lights ----
        n_aux = self.cfg["aux_lights"]
        aux_energy = self.cfg["aux_light_energy"]
        positions = self._aux_positions(n_aux, lx, ly, lz)
        for i, pos in enumerate(positions):
            self._add_light(
                name=f"AuxLight_{i:02d}",
                light_type="POINT",
                location=pos,
                energy=aux_energy,
                color=(0.9, 0.95, 1.0)       # slightly cool fill
            )

        print(f"[LightingSetup] 1 main + {n_aux} aux lights added.")

    # ------------------------------------------------------------------ #
    def _remove_existing_lights(self):
        for obj in bpy.data.objects:
            if obj.type == "LIGHT":
                bpy.data.objects.remove(obj, do_unlink=True)

    def _add_light(self, name: str, light_type: str, location: tuple,
                   energy: float, color: tuple = (1, 1, 1), size: float = 1.0):
        light_data = bpy.data.lights.new(name=name, type=light_type)
        light_data.energy = energy
        light_data.color = color
        if light_type == "AREA":
            light_data.shape = "SQUARE"
            light_data.size = size
        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        bpy.context.collection.objects.link(light_obj)
        light_obj.location = location
        return light_obj

    def _aux_positions(self, n: int, lx: float, ly: float, lz: float) -> List[tuple]:
        """Distribute n aux lights evenly across the room at mid-height."""
        h = lz * 0.75
        if n == 0:
            return []
        if n == 1:
            return [(lx * 0.3, ly * 0.3, h)]
        # Distribute on a simple grid pattern
        positions = []
        xs = [-lx * 0.3, lx * 0.3]
        ys = [-ly * 0.3, ly * 0.3]
        for i in range(n):
            positions.append((xs[i % 2], ys[(i // 2) % 2], h))
        return positions[:n]
