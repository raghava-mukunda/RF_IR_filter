"""
CliQ10 iteration baseline using the previously successful Run 013 HFSS pattern.

This creates a NEW project. It intentionally reuses the old working pattern:
- internal air box spanning x=0..L
- subtract copper conductor sections from the air box with keep_originals=True
- explicitly set air.solve_inside=True
- wave-port sheets on the internal cavity cross-section
- hfss.wave_port(...) WITHOUT create_pec_cap=True
- setup.create_frequency_sweep(...) as in the working build_filter.py

This is an optimization baseline, not an exact SCALINQ CliQ10 reconstruction.
It has no hollow absorber channels yet. After a successful solve, iterate
geometry parameters and compare S11/S21 with the datasheet.

Run from repository root:
    .\\.venv\\Scripts\\python.exe hfss\\build_cliq10_iter01.py
"""

from pathlib import Path
import json
import traceback

from ansys.aedt.core import Desktop, Hfss

# ---------------- Project: new and separate ----------------
OUT = Path("hfss") / "cliq10_iter01"
PROJECT_NAME = "CliQ10_Iter01"
DESIGN_NAME = "CliQ10_Iter01"
AEDT_VERSION = "2025.2"

# ---------------- Package / enclosure dimensions (mm) ----------------
L = 65.3
OUTER_W = 5.0
OUTER_H = 4.6
SIDE_WALL = 0.20
TOP_BOTTOM_WALL = 0.20

INNER_W = OUTER_W - 2 * SIDE_WALL       # 4.6 mm
INNER_H = OUTER_H - 2 * TOP_BOTTOM_WALL # 4.2 mm

# ---------------- Iteration parameters (mm) ----------------
# Initial optimization variables, not asserted proprietary CliQ10 dimensions.
N_SECTIONS = 9
FEED_LENGTH = 4.0
STRIP_WIDTH = 1.20
FEED_THICKNESS = 0.20
THIN_THICKNESS = 0.12
THICK_THICKNESS = 0.35
SECTION_WEIGHTS = [1.0, 1.2, 1.5, 1.8, 2.0, 1.8, 1.5, 1.2, 1.0]

# ---------------- HFSS setup ----------------
F_ADAPT = "40GHz"
F_START_GHZ = 0.01
F_STOP_GHZ = 40.0
MAX_PASSES = 10
DELTA_S = 0.02
PORT_MODES = 2


def require(obj, label):
    if obj is None or obj is False:
        raise RuntimeError(f"AEDT failed to create {label}")
    return obj


def create_box(hfss, name, origin, sizes, material):
    return require(
        hfss.modeler.create_box(
            origin=origin, sizes=sizes, name=name, material=material
        ),
        name,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    project_path = (OUT / f"{PROJECT_NAME}.aedt").resolve()
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

        # 1) Enclosure walls: same style as the known working Run 013 model.
        # The four walls do not overlap each other at the corners.
        create_box(
            hfss, "WallBottom",
            [0, -OUTER_W / 2, -OUTER_H / 2],
            [L, OUTER_W, TOP_BOTTOM_WALL], "copper"
        )
        create_box(
            hfss, "WallTop",
            [0, -OUTER_W / 2, OUTER_H / 2 - TOP_BOTTOM_WALL],
            [L, OUTER_W, TOP_BOTTOM_WALL], "copper"
        )
        create_box(
            hfss, "WallYMinus",
            [0, -OUTER_W / 2, -OUTER_H / 2 + TOP_BOTTOM_WALL],
            [L, SIDE_WALL, OUTER_H - 2 * TOP_BOTTOM_WALL], "copper"
        )
        create_box(
            hfss, "WallYPlus",
            [0, OUTER_W / 2 - SIDE_WALL, -OUTER_H / 2 + TOP_BOTTOM_WALL],
            [L, SIDE_WALL, OUTER_H - 2 * TOP_BOTTOM_WALL], "copper"
        )

        # 2) Stepped centre conductor: continuous axial sections with no overlap.
        # First and last feeds are part of the section sequence so each segment
        # has a unique x interval and shares only a face with its neighbour.
        active_length = L - 2 * FEED_LENGTH
        total_weight = sum(SECTION_WEIGHTS)
        section_lengths = [
            active_length * weight / total_weight for weight in SECTION_WEIGHTS
        ]

        conductor_objects = []
        section_table = []

        conductor_objects.append(create_box(
            hfss, "InputFeed",
            [0, -STRIP_WIDTH / 2, -FEED_THICKNESS / 2],
            [FEED_LENGTH, STRIP_WIDTH, FEED_THICKNESS], "copper"
        ))

        x = FEED_LENGTH
        for i, seg_len in enumerate(section_lengths):
            thickness = THIN_THICKNESS if i % 2 == 0 else THICK_THICKNESS
            name = f"CoreSection{i + 1:02d}"
            obj = create_box(
                hfss, name,
                [x, -STRIP_WIDTH / 2, -thickness / 2],
                [seg_len, STRIP_WIDTH, thickness], "copper"
            )
            conductor_objects.append(obj)
            section_table.append({
                "name": name,
                "x_start_mm": x,
                "length_mm": seg_len,
                "thickness_mm": thickness,
            })
            x += seg_len

        conductor_objects.append(create_box(
            hfss, "OutputFeed",
            [L - FEED_LENGTH, -STRIP_WIDTH / 2, -FEED_THICKNESS / 2],
            [FEED_LENGTH, STRIP_WIDTH, FEED_THICKNESS], "copper"
        ))

        # 3) Internal air volume. This mirrors the old successful script:
        # air occupies only the enclosure interior, not a padded external region.
        air = create_box(
            hfss, "AirInterior",
            [0, -INNER_W / 2, -INNER_H / 2],
            [L, INNER_W, INNER_H], "air"
        )

        subtract_result = hfss.modeler.subtract(
            blank_list=air,
            tool_list=conductor_objects,
            keep_originals=True,
        )
        if subtract_result is False:
            raise RuntimeError(
                "Subtracting centre-conductor solids from AirInterior failed."
            )
        air.solve_inside = True

        # 4) End-port sheets: use the exact interior aperture cross-section,
        # matching the proven Run 013 implementation. No auto PEC cap.
        port1_sheet = require(
            hfss.modeler.create_rectangle(
                orientation="YZ",
                origin=[0, -INNER_W / 2, -INNER_H / 2],
                sizes=[INNER_W, INNER_H],
                name="Port1Sheet",
            ),
            "Port1Sheet",
        )
        port2_sheet = require(
            hfss.modeler.create_rectangle(
                orientation="YZ",
                origin=[L, -INNER_W / 2, -INNER_H / 2],
                sizes=[INNER_W, INNER_H],
                name="Port2Sheet",
            ),
            "Port2Sheet",
        )

        p1 = hfss.wave_port(
            assignment=port1_sheet.name,
            name="Port1",
            modes=PORT_MODES,
            impedance=50,
            renormalize=True,
        )
        p2 = hfss.wave_port(
            assignment=port2_sheet.name,
            name="Port2",
            modes=PORT_MODES,
            impedance=50,
            renormalize=True,
        )
        require(p1, "Port1")
        require(p2, "Port2")

        # 5) Reuse the working setup/sweep API from build_filter.py.
        setup = hfss.create_setup("Setup1")
        setup.props["Frequency"] = F_ADAPT
        setup.props["MaximumPasses"] = MAX_PASSES
        setup.props["MinimumPasses"] = 2
        setup.props["MinimumConvergedPasses"] = 1
        setup.props["DeltaS"] = DELTA_S
        if not setup.update():
            raise RuntimeError("Failed to update Setup1")

        sweep = setup.create_frequency_sweep(
            unit="GHz",
            name="Sweep1",
            start_frequency=F_START_GHZ,
            stop_frequency=F_STOP_GHZ,
            sweep_type="Interpolating",
            save_fields=False,
        )
        require(sweep, "Sweep1")

        parameters = {
            "status": "ITERATION 01 BASELINE - NOT YET CLIQ10 VALIDATED",
            "project": str(project_path),
            "envelope_mm": [L, OUTER_W, OUTER_H],
            "inner_aperture_mm": [INNER_W, INNER_H],
            "wall_thickness_mm": [SIDE_WALL, TOP_BOTTOM_WALL],
            "strip_width_mm": STRIP_WIDTH,
            "feed_length_mm": FEED_LENGTH,
            "feed_thickness_mm": FEED_THICKNESS,
            "thin_thickness_mm": THIN_THICKNESS,
            "thick_thickness_mm": THICK_THICKNESS,
            "sections": section_table,
            "sweep_GHz": [F_START_GHZ, F_STOP_GHZ],
            "known_missing": [
                "Hollow absorber channels and terminations",
                "Exact CliQ10 internal geometry",
                "S-parameter optimization against datasheet",
            ],
        }
        (OUT / "parameters_iter01.json").write_text(
            json.dumps(parameters, indent=2), encoding="utf-8"
        )
        hfss.save_project()

        print(f"\nCreated project: {project_path}")
        print("Using the known working Run 013 pattern for internal air and ports.")
        print("Starting Setup1 solve...")
        result = hfss.analyze_setup("Setup1")
        print(f"analyze_setup('Setup1') returned: {result}")
        hfss.save_project()

        if not result:
            print(
                "SOLVE FAILED. Send the first red AEDT Message Manager error. "
                "This is a fresh project; Run 013 remains untouched."
            )
            input("Press Enter to close.")
            return

        print("HFSS reports a successful solve. Creating S-parameter report...")
        try:
            report = hfss.post.create_report(
                expressions=["dB(S(Port1,Port1))", "dB(S(Port2,Port1))"],
                setup_sweep_name="Setup1 : Sweep1",
                primary_sweep_variable="Freq",
                report_category="Standard",
                plot_type="Rectangular Plot",
            )
            if report:
                csv_path = (OUT / "sparams_iter01.csv").resolve()
                plot_name = getattr(report, "plot_name", report)
                hfss.post.export_report_to_csv(
                    output_file=str(csv_path),
                    plot_name=plot_name,
                )
                print(f"Report export requested: {csv_path}")
            else:
                print("Report creation returned no report; project is saved.")
        except Exception as exc:
            print(f"Report export issue (solve remains saved): {exc}")

        print(
            "This is an S-parameter baseline only. It is not yet a CliQ10 "
            "performance match because hollow absorbers are not implemented."
        )
        input("Press Enter to close.")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if desktop is not None:
            try:
                desktop.release_desktop(close_projects=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()
