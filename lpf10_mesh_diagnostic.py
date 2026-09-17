import os
from pathlib import Path
import numpy as np

# ============================================================
# YOUR ACTUAL openEMS INSTALLATION
# ============================================================
OPENEMS_ROOT = Path(
    r"C:\Users\Raghava.m\Desktop\openEMS_x64_v0.0.36-93-g7b9cd51_msvc\openEMS"
)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(OPENEMS_ROOT))

os.environ["CSXCAD_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["OPENEMS_INSTALL_PATH"] = str(OPENEMS_ROOT)
os.environ["PATH"] = (
    str(OPENEMS_ROOT)
    + os.pathsep
    + os.environ.get("PATH", "")
)

from CSXCAD import ContinuousStructure


# ============================================================
# SETTINGS
# ============================================================
UNIT = 1e-3

# V2 optimized 21-section lengths
L_MM = [
    1.5490741064594331,
    2.9993952714422627,
    2.6584447429668714,
    3.2336204730268134,
    2.6254572534258034,
    2.8772944265839335,
    1.8542308138965293,
    1.590614543218636,
    1.6549781916882556,
    1.6411637418873382,
    1.3959595952069717,
    1.6411637418873382,
    1.6549781916882556,
    1.590614543218636,
    1.8542308138965293,
    2.8772944265839335,
    2.6254572534258034,
    3.2336204730268134,
    2.6584447429668714,
    2.9993952714422627,
    1.5490741064594331,
]

FEED_LENGTH = 10.0

FEED_W = 3.0
FEED_H = 0.50

LOW_W = 1.5
LOW_H = 0.50

HIGH_W = 3.0
HIGH_H = 1.45

CENTER_W = 0.8
CENTER_T = 0.30

WALL_T = 0.20

DX = 1.0
DY = 1.0
DZ = 1.0


# ============================================================
# HELPERS
# ============================================================
def box(prop, p1, p2, priority):
    prop.AddBox(
        np.array(p1, dtype=float),
        np.array(p2, dtype=float),
        priority=priority,
    )


# ============================================================
# BUILD
# ============================================================
def build():

    print()
    print("==============================================")
    print("OPENEMS / CSXCAD MESH DIAGNOSTIC")
    print("==============================================")
    print("Building geometry...")
    print()

    CSX = ContinuousStructure()
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(UNIT)

    # ONLY materials needed for this diagnostic.
    PEC = CSX.AddMetal("PEC")
    AIR = CSX.AddMaterial("AIR", epsilon=1.0)

    filter_length = sum(L_MM)

    x0 = -FEED_LENGTH
    x1 = filter_length + FEED_LENGTH

    print(f"Filter length : {filter_length:.6f} mm")
    print(f"Simulation X  : {x0:.6f} -> {x1:.6f} mm")

    # ========================================================
    # FEED + 21 SECTIONS
    # ========================================================
    x = 0.0

    section_edges = [0.0]

    for i, length in enumerate(L_MM):

        if i % 2 == 0:
            W = HIGH_W
            H = HIGH_H
        else:
            W = LOW_W
            H = LOW_H

        xa = x
        xb = x + length

        # -----------------------------
        # Center conductor
        # -----------------------------
        box(
            PEC,
            [
                xa,
                -CENTER_W / 2,
                -CENTER_T / 2,
            ],
            [
                xb,
                CENTER_W / 2,
                CENTER_T / 2,
            ],
            priority=30,
        )

        # -----------------------------
        # Top wall
        # -----------------------------
        box(
            PEC,
            [
                xa,
                -W / 2,
                H / 2 - WALL_T,
            ],
            [
                xb,
                W / 2,
                H / 2,
            ],
            priority=10,
        )

        # -----------------------------
        # Bottom wall
        # -----------------------------
        box(
            PEC,
            [
                xa,
                -W / 2,
                -H / 2,
            ],
            [
                xb,
                W / 2,
                -H / 2 + WALL_T,
            ],
            priority=10,
        )

        # -----------------------------
        # Left wall
        # -----------------------------
        box(
            PEC,
            [
                xa,
                -W / 2,
                -H / 2,
            ],
            [
                xb,
                -W / 2 + WALL_T,
                H / 2,
            ],
            priority=10,
        )

        # -----------------------------
        # Right wall
        # -----------------------------
        box(
            PEC,
            [
                xa,
                W / 2 - WALL_T,
                -H / 2,
            ],
            [
                xb,
                W / 2,
                H / 2,
            ],
            priority=10,
        )

        x = xb
        section_edges.append(x)

    # ========================================================
    # FEED CENTER CONDUCTORS
    # ========================================================
    box(
        PEC,
        [
            x0,
            -CENTER_W / 2,
            -CENTER_T / 2,
        ],
        [
            0.0,
            CENTER_W / 2,
            CENTER_T / 2,
        ],
        priority=30,
    )

    box(
        PEC,
        [
            filter_length,
            -CENTER_W / 2,
            -CENTER_T / 2,
        ],
        [
            x1,
            CENTER_W / 2,
            CENTER_T / 2,
        ],
        priority=30,
    )

    # ========================================================
    # FEED WALLS
    # ========================================================
    for xa, xb in [(x0, 0.0), (filter_length, x1)]:

        # top
        box(
            PEC,
            [
                xa,
                -FEED_W / 2,
                FEED_H / 2 - WALL_T,
            ],
            [
                xb,
                FEED_W / 2,
                FEED_H / 2,
            ],
            priority=10,
        )

        # bottom
        box(
            PEC,
            [
                xa,
                -FEED_W / 2,
                -FEED_H / 2,
            ],
            [
                xb,
                FEED_W / 2,
                -FEED_H / 2 + WALL_T,
            ],
            priority=10,
        )

        # left
        box(
            PEC,
            [
                xa,
                -FEED_W / 2,
                -FEED_H / 2,
            ],
            [
                xb,
                -FEED_W / 2 + WALL_T,
                FEED_H / 2,
            ],
            priority=10,
        )

        # right
        box(
            PEC,
            [
                xa,
                FEED_W / 2 - WALL_T,
                -FEED_H / 2,
            ],
            [
                xb,
                FEED_W / 2,
                FEED_H / 2,
            ],
            priority=10,
        )

    # ========================================================
    # MESH
    # ========================================================

    # X mesh
    x_lines = set()

    x_lines.add(x0)
    x_lines.add(0.0)
    x_lines.add(filter_length)
    x_lines.add(x1)

    for v in section_edges:
        x_lines.add(v)

    # Add section centers
    for i in range(len(section_edges) - 1):
        xc = (
            section_edges[i]
            + section_edges[i + 1]
        ) / 2.0

        x_lines.add(xc)

    # Y mesh
    y_lines = set(
        np.arange(
            -2.0,
            2.001,
            DY
        )
    )

    y_lines.update([
        -FEED_W / 2,
        -HIGH_W / 2,
        -LOW_W / 2,
        -CENTER_W / 2,
        0.0,
        CENTER_W / 2,
        LOW_W / 2,
        HIGH_W / 2,
        FEED_W / 2,
    ])

    # Z mesh
    z_lines = set(
        np.arange(
            -4.0,
            4.001,
            DZ
        )
    )

    z_lines.update([
        -HIGH_H / 2,
        -LOW_H / 2,
        -FEED_H / 2,
        -CENTER_T / 2,
        0.0,
        CENTER_T / 2,
        FEED_H / 2,
        LOW_H / 2,
        HIGH_H / 2,
    ])

    mesh.AddLine(
        "x",
        sorted(x_lines)
    )

    mesh.AddLine(
        "y",
        sorted(y_lines)
    )

    mesh.AddLine(
        "z",
        sorted(z_lines)
    )

    # ========================================================
    # CENSUS
    # ========================================================
    print()
    print("==============================================")
    print("ACTUAL CSXCAD MESH")
    print("==============================================")

    for axis in ["x", "y", "z"]:

        try:
            lines = np.asarray(
                mesh.GetLines(axis),
                dtype=float
            ).reshape(-1)

            delta = np.diff(lines)

            print()
            print(
                f"{axis.upper()} AXIS"
            )
            print(
                f"  lines       = {len(lines)}"
            )
            print(
                f"  cells       = {len(delta)}"
            )
            print(
                f"  minimum     = {delta.min():.12g} mm"
            )
            print(
                f"  maximum     = {delta.max():.12g} mm"
            )

            idx = np.argmin(delta)

            print(
                f"  smallest cell:"
            )
            print(
                f"    {lines[idx]:.12g}"
                f" -> "
                f"{lines[idx + 1]:.12g} mm"
            )

        except Exception as e:
            print(
                f"{axis.upper()} ERROR: {e}"
            )

    print()
    print("==============================================")
    print("TOTAL CARTESIAN CELLS")
    print("==============================================")

    nx = len(
        np.asarray(
            mesh.GetLines("x")
        ).reshape(-1)
    ) - 1

    ny = len(
        np.asarray(
            mesh.GetLines("y")
        ).reshape(-1)
    ) - 1

    nz = len(
        np.asarray(
            mesh.GetLines("z")
        ).reshape(-1)
    ) - 1

    print(f"NX = {nx}")
    print(f"NY = {ny}")
    print(f"NZ = {nz}")
    print(f"TOTAL = {nx * ny * nz:,}")

    print()
    print("==============================================")
    print("DIAGNOSTIC COMPLETE")
    print("NO FDTD OPERATOR CREATED")
    print("==============================================")


if __name__ == "__main__":
    build()