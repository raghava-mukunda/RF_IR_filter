"""
CliQ10-inspired HFSS candidate builder
======================================

IMPORTANT:
- This is an independent, geometry-level candidate, NOT an exact reconstruction
  of SCALINQ's proprietary CliQ10 internals and NOT a validated filter.
- It uses the architecture described in the uploaded 2025 paper:
  stepped-impedance rectangular coaxial line + hollow absorber channels.
- The paper's optimized dimensions/section lengths are used as a starting point.
  They were optimized for a different prototype, so the CliQ10 datasheet targets
  are NOT guaranteed.
- Inspect geometry, port modes, and Boolean operations before solving.
- This script creates a NEW AEDT project; it does not modify Run 013.

Run from repository root:
    .\\.venv\\Scripts\\python.exe hfss\\build_herd_filter.py

PyAEDT 1.9.x / AEDT 2025 R2 compatible starting point.
"""

from pathlib import Path
import json
import math
import traceback

from ansys.aedt.core import Desktop, Hfss

# -------------------------- USER SETTINGS --------------------------
PROJECT_NAME = "CliQ10_Inspired_Candidate"
DESIGN_NAME = "CliQ10_Candidate"
OUTPUT_DIR = Path("hfss") / "cliq10_candidate"

# Published CliQ10 SMPM external envelope (datasheet)
LENGTH = 65.3
WIDTH = 5.0
HEIGHT = 4.6

# 2025 paper prototype parameters (not CliQ10 proprietary dimensions)
# Two alternating rectangular-coax outer cross sections:
# low impedance:  wo=3.0 mm, ho=1.45 mm
# high impedance: wo=1.5 mm, ho=0.50 mm
WO_LOW = 3.0
HO_LOW = 1.45
WO_HIGH = 1.5
HO_HIGH = 0.50
HI_LOW = 0.30
HI_HIGH = 0.80
STRIP_THICKNESS = 0.50  # paper describes a 0.5 mm copper strip; used as thickness

# Published table II has 11 optimized section lengths (mm). These are from the
# 2025 paper, NOT optimized for the CliQ10 datasheet.
HALF_SECTION_LENGTHS = [1.06, 2.30, 2.05, 2.54, 1.91, 3.16, 1.91, 4.28, 1.77, 4.63, 0.80]

# Hollow waveguide dimensions from the 2025 paper
HW_RADIUS_SMALL = 1.15
HW_RADIUS_LARGE = 1.50
HW_DEPTH = 2.50
# The article says MACOR inserts have epsilon_r=5.64 and tan(delta)=2.5e-3.
# A material plug is created as a simple insert candidate; its exact shape/fit
# and absorption termination are not fully specified by the paper geometry.
MACOR_ER = 5.64
MACOR_TAND = 2.5e-3

F_ADAPT = "40GHz"
F_START_GHZ = 0.01
F_STOP_GHZ = 70.0
F_STEP_GHZ = 0.25
MAX_PASSES = 12
DELTA_S = 0.02
PORT_MODES = 2

# Default False: build/save and inspect first. Set True only after inspection.
SOLVE_AFTER_BUILD = False
AEDT_VERSION = "2025.2"
# ------------------------------------------------------------------


def checked(obj, label):
    if obj is None or obj is False:
        raise RuntimeError(f"AEDT failed to create: {label}")
    return obj


def safe_material(hfss, name, eps_r, tan_delta):
    """Create a dielectric material if not already present."""
    try:
        mat = hfss.materials[name]
    except Exception:
        mat = None
    if mat is None:
        mat = hfss.materials.add_material(name)
    mat.permittivity = eps_r
    mat.dielectric_loss_tangent = tan_delta
    return mat


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    project_path = (OUTPUT_DIR / f"{PROJECT_NAME}.aedt").resolve()

    desktop = None
    hfss = None
    try:
        desktop = Desktop(
            version=AEDT_VERSION,
            student_version=True,
            non_graphical=False,
            new_desktop=True,
        )
        hfss = Hfss(
            project=str(project_path),
            design=DESIGN_NAME,
            solution_type="Modal",
            version=AEDT_VERSION,
            new_desktop=False,
        )
        hfss.modeler.model_units = "mm"

        # Keep material names distinct from any existing project defaults.
        safe_material(hfss, "CliQ_MACOR", MACOR_ER, MACOR_TAND)

        # Construct the enclosure as separate copper wall solids instead of
        # subtracting a stack of touching air boxes from a bulk solid. This avoids
        # the invalid Boolean subtraction seen in AEDT. The interior is air by
        # default; do not create overlapping "air cavity" solids.
        half_w = WIDTH / 2
        half_h = HEIGHT / 2
        wall_t = 0.25

        wall_specs = [
            ("WallBottom", [0, -half_w, -half_h], [LENGTH, WIDTH, wall_t]),
            ("WallTop", [0, -half_w, half_h - wall_t], [LENGTH, WIDTH, wall_t]),
            ("WallYMinus", [0, -half_w, -half_h + wall_t],
             [LENGTH, wall_t, HEIGHT - 2 * wall_t]),
            ("WallYPlus", [0, half_w - wall_t, -half_h + wall_t],
             [LENGTH, wall_t, HEIGHT - 2 * wall_t]),
        ]
        wall_objects = []
        for name, origin, sizes in wall_specs:
            wall_objects.append(checked(
                hfss.modeler.create_box(
                    origin=origin, sizes=sizes, name=name, material="copper"
                ),
                name,
            ))

        # The research-paper stepped section lengths are scaled to fit the
        # requested external length. This is a packaging approximation only.
        section_lengths = HALF_SECTION_LENGTHS + list(reversed(HALF_SECTION_LENGTHS[:-1]))
        if len(section_lengths) != 21:
            raise RuntimeError("Expected 21 stepped-impedance sections.")
        total_active = sum(section_lengths)
        active_target = LENGTH - 2.0
        scale = active_target / total_active
        section_lengths = [v * scale for v in section_lengths]
        total_active = sum(section_lengths)
        x0 = (LENGTH - total_active) / 2

        section_info = []
        x = x0
        for i, seg_len in enumerate(section_lengths):
            high_z = (i % 2 == 0)
            section_info.append({
                "section": i + 1,
                "x_start_mm": x,
                "length_mm": seg_len,
                "impedance_section": "high" if high_z else "low",
                "outer_width_parameter_mm": WO_HIGH if high_z else WO_LOW,
                "outer_height_parameter_mm": HO_HIGH if high_z else HO_LOW,
                "inner_strip_height_parameter_mm": HI_HIGH if high_z else HI_LOW,
            })
            x += seg_len

        # Create the centre conductor as section solids. These are intentionally
        # kept as separate objects for inspectability; their end faces meet at
        # section boundaries. Check in AEDT that the junctions are contiguous.
        strip_parts = []
        x = x0
        for i, seg_len in enumerate(section_lengths):
            high_z = (i % 2 == 0)
            thickness = HI_HIGH if high_z else HI_LOW
            strip = checked(
                hfss.modeler.create_box(
                    origin=[x, -STRIP_THICKNESS / 2, -thickness / 2],
                    sizes=[seg_len, STRIP_THICKNESS, thickness],
                    name=f"CenterConductor_{i+1:02d}",
                    material="copper",
                ),
                f"CenterConductor_{i+1:02d}",
            )
            strip_parts.append(strip)
            x += seg_len

        # Hollow absorber channels are deliberately NOT auto-cut in this
        # recovery version. The previous implementation failed AEDT Boolean
        # subtraction. Cutting through wall solids robustly requires a verified
        # CAD topology; do not represent standalone air cylinders as working
        # absorber channels. This model is therefore a stepped-line/enclosure
        # geometry scaffold, not a complete CliQ10-inspired filter.
        print(
            "NOTICE: hollow absorber channels are not built in this recovery "
            "script; do not interpret this as a complete CliQ10 model."
        )

        # Ports: use full cross-section sheets at both ends, with integration
        # line along z from centre conductor toward outer conductor.
        port1_sheet = checked(
            hfss.modeler.create_rectangle(
                orientation="YZ",
                origin=[0, -WIDTH / 2, -HEIGHT / 2],
                sizes=[WIDTH, HEIGHT],
                name="PortSheet1",
            ),
            "PortSheet1",
        )
        port2_sheet = checked(
            hfss.modeler.create_rectangle(
                orientation="YZ",
                origin=[LENGTH, -WIDTH / 2, -HEIGHT / 2],
                sizes=[WIDTH, HEIGHT],
                name="PortSheet2",
            ),
            "PortSheet2",
        )
        p1 = hfss.wave_port(
            port1_sheet,
            modes=PORT_MODES,
            impedance=50,
            renormalize=True,
            name="Port1",
        )
        p2 = hfss.wave_port(
            port2_sheet,
            modes=PORT_MODES,
            impedance=50,
            renormalize=True,
            name="Port2",
        )
        checked(p1, "Port1 wave port")
        checked(p2, "Port2 wave port")

        # Solution setup and wide sweep.
        setup = hfss.create_setup("Setup1")
        setup.props["Frequency"] = F_ADAPT
        setup.props["MaximumPasses"] = MAX_PASSES
        setup.props["MinimumPasses"] = 2
        setup.props["MinimumConvergedPasses"] = 1
        setup.props["DeltaS"] = DELTA_S
        setup.update()

        number_of_points = int(round((F_STOP_GHZ - F_START_GHZ) / F_STEP_GHZ)) + 1
        sweep = hfss.create_linear_count_sweep(
            setup="Setup1",
            unit="GHz",
            start_frequency=F_START_GHZ,
            stop_frequency=F_STOP_GHZ,
            num_of_freq_points=number_of_points,
            sweep_type="Interpolating",
            name="Sweep1",
            save_fields=False,
        )
        if sweep:
            sweep.props["SaveFields"] = False
            sweep.update()

        params = {
            "project": PROJECT_NAME,
            "design": DESIGN_NAME,
            "status": "UNVALIDATED CANDIDATE",
            "purpose": "Geometry scaffold only; not an exact reproduction or complete filter",
            "known_missing": [
                "Hollow absorber channels and absorber terminations are not implemented in this recovery version",
                "Geometry has not been validated against the CliQ10 internal construction",
                "Do not treat as a CliQ10 reproduction or manufacturing-ready design",
            ],
            "envelope_mm": [LENGTH, WIDTH, HEIGHT],
            "reference_impedance_ohm": 50,
            "source_geometry": "2025 co-designed reflective/leaky-waveguide paper; not CliQ10 internal geometry",
            "source_section_lengths_mm_before_packaging_scale": HALF_SECTION_LENGTHS,
            "axial_length_scale_applied": scale,
            "section_lengths_mm_after_packaging_scale": section_lengths,
            "sections": section_info,
            "waveguide": {
                "radii_mm": [HW_RADIUS_SMALL, HW_RADIUS_LARGE],
                "depth_mm": HW_DEPTH,
                "note": "Simplified bores; full absorber/termination geometry not disclosed in CliQ10 datasheet",
            },
            "datasheet_targets_not_yet_verified": {
                "passband_10mK": "DC-9.8 GHz, 3 dB criterion",
                "insertion_loss_10mK": "<=0.5 dB DC-7.5 GHz; <=1.5 dB 7.5-9.3 GHz",
                "return_loss_298K": ">15 dB up to 9.3 GHz",
                "rejection_298K": ">=30 dB from 11 GHz; >=55 dB from 55 GHz",
            },
        }
        (OUTPUT_DIR / "parameters.json").write_text(
            json.dumps(params, indent=2), encoding="utf-8"
        )

        hfss.save_project()
        print("\nCandidate geometry created and saved.")
        print(f"Project: {project_path}")
        print(f"Parameters: {(OUTPUT_DIR / 'parameters.json').resolve()}")
        print("STATUS: NOT VALIDATED; inspect geometry, Boolean results, and ports.")
        print("Do not treat this model as manufacturing-ready.")
        if SOLVE_AFTER_BUILD:
            print("Starting HFSS analysis...")
            ok = hfss.analyze_setup("Setup1")
            print(f"HFSS analyze_setup returned: {ok}")
            hfss.save_project()
            if not ok:
                print("SOLVE FAILED. Inspect AEDT Messages and solver profile/log.")
        else:
            print("SOLVE_AFTER_BUILD=False; inspect geometry before enabling solve.")
            input("Press Enter to close AEDT, or Ctrl+C to leave it open.")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        # Do not forcibly close other user projects.
        if desktop is not None:
            try:
                desktop.release_desktop(close_projects=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()
