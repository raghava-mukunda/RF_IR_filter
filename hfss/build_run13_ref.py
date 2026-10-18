"""
Run-013 geometry reconstruction for HFSS (PyAEDT 1.9.x / AEDT 2025 R2).

Creates a NEW project only. It does NOT open, edit, or overwrite Run 013,
CliQ10_Iter01, or any optimiser evaluation.

Geometry is transcribed from the archived openEMS Run-013 script:
  enclosure 65.3 x 5.0 x 4.6 mm; 0.2 mm walls
  centre strip width 1.2 mm
  8 mm input/output launches
  h_thin = 0.28671928 mm; h_thick = 0.76349166 mm
  7 symmetric section lengths from Run 013
  lumped-port planes at x=7.0 and x=58.3 mm, with 50-ohm ports.

IMPORTANT:
HFSS and openEMS use different solvers/port implementations. This is a
geometry-and-port reconstruction for a controlled cross-check, not a claim
that the S-parameters will match without validation. Review port assignment
and AEDT messages before trusting results.
"""

from pathlib import Path
import json
import traceback

from ansys.aedt.core import Desktop, Hfss

# ---------- Isolated output: no existing project is overwritten ----------
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "run13_hfss_reference"
PROJECT = OUT / "Run13_HFSS_Reference.aedt"
DESIGN = "Run13_HFSS_Reference"
AEDT_VERSION = "2025.2"

# ---------- Exact archived Run-013 geometry, mm ----------
TOTAL_LENGTH = 65.3
TOTAL_WIDTH = 5.0
TOTAL_HEIGHT = 4.6
WALL = 0.2
CENTER_WIDTH = 1.2
INPUT_LAUNCH = 8.0
OUTPUT_LAUNCH = 8.0
H_THIN = 0.28671928
H_THICK = 0.76349166
SECTION_LENGTHS = [
    4.07365691,
    7.53238661,
    8.07101208,
    9.94588880,
    8.07101208,
    7.53238661,
    4.07365691,
]

# Archived openEMS port planes and target impedance
PORT1_X = INPUT_LAUNCH - 1.0            # 7.0 mm
PORT2_X = TOTAL_LENGTH - OUTPUT_LAUNCH + 1.0  # 58.3 mm
Z0 = 50.0

# HFSS solve configuration; intentionally modest for first cross-check.
F_ADAPT = "28GHz"
F_START_GHZ = 0.5
F_STOP_GHZ = 28.0
MAX_PASSES = 10
DELTA_S = 0.02


def need(value, what):
    if value is None or value is False:
        raise RuntimeError(f"AEDT failed to create/update {what}")
    return value


def create_box(hfss, name, origin, sizes, material):
    return need(
        hfss.modeler.create_box(
            origin=origin, sizes=sizes, name=name, material=material
        ),
        name,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    desktop = None
    hfss = None

    # Hard stop if this exact output already exists; never overwrite it.
    if PROJECT.exists():
        raise FileExistsError(
            f"{PROJECT} already exists. Rename/move it yourself before rerunning; "
            "this script will not overwrite an existing project."
        )

    try:
        desktop = Desktop(
            version=AEDT_VERSION,
            student_version=True,
            non_graphical=False,
            new_desktop=True,
        )
        hfss = Hfss(
            project=str(PROJECT),
            design=DESIGN,
            solution_type="Modal",
            version=AEDT_VERSION,
            new_desktop=False,
        )
        hfss.modeler.model_units = "mm"

        # Enclosure is represented as four non-overlapping copper walls.
        create_box(
            hfss, "WallBottom",
            [0, -TOTAL_WIDTH/2, -TOTAL_HEIGHT/2],
            [TOTAL_LENGTH, TOTAL_WIDTH, WALL], "copper"
        )
        create_box(
            hfss, "WallTop",
            [0, -TOTAL_WIDTH/2, TOTAL_HEIGHT/2 - WALL],
            [TOTAL_LENGTH, TOTAL_WIDTH, WALL], "copper"
        )
        create_box(
            hfss, "WallYMinus",
            [0, -TOTAL_WIDTH/2, -TOTAL_HEIGHT/2 + WALL],
            [TOTAL_LENGTH, WALL, TOTAL_HEIGHT - 2*WALL], "copper"
        )
        create_box(
            hfss, "WallYPlus",
            [0, TOTAL_WIDTH/2 - WALL, -TOTAL_HEIGHT/2 + WALL],
            [TOTAL_LENGTH, WALL, TOTAL_HEIGHT - 2*WALL], "copper"
        )

        # Continuous centre conductor. Seven central sections alternate
        # thin/thick/thin/thick/thin/thick/thin and touch face-to-face.
        conductor = []
        conductor.append(create_box(
            hfss, "InputLaunch",
            [0, -CENTER_WIDTH/2, -H_THIN/2],
            [INPUT_LAUNCH, CENTER_WIDTH, H_THIN], "copper"
        ))

        x = INPUT_LAUNCH
        section_records = []
        for i, length in enumerate(SECTION_LENGTHS):
            height = H_THIN if i % 2 == 0 else H_THICK
            obj = create_box(
                hfss, f"Section_{i+1}",
                [x, -CENTER_WIDTH/2, -height/2],
                [length, CENTER_WIDTH, height], "copper"
            )
            conductor.append(obj)
            section_records.append({
                "section": i + 1,
                "x_start_mm": x,
                "x_end_mm": x + length,
                "length_mm": length,
                "height_mm": height,
            })
            x += length

        conductor.append(create_box(
            hfss, "OutputLaunch",
            [TOTAL_LENGTH - OUTPUT_LAUNCH, -CENTER_WIDTH/2, -H_THIN/2],
            [OUTPUT_LAUNCH, CENTER_WIDTH, H_THIN], "copper"
        ))

        expected_x = TOTAL_LENGTH - OUTPUT_LAUNCH
        if abs(x - expected_x) > 1e-6:
            raise RuntimeError(
                f"Section-length sum mismatch: last section ends at {x:.9f} mm, "
                f"expected {expected_x:.9f} mm"
            )

        # Interior air is the volume between the four walls. Subtract the
        # centre conductor so the volume is not overlapping metal.
        air = create_box(
            hfss, "AirInterior",
            [0, -TOTAL_WIDTH/2 + WALL, -TOTAL_HEIGHT/2 + WALL],
            [TOTAL_LENGTH, TOTAL_WIDTH - 2*WALL, TOTAL_HEIGHT - 2*WALL],
            "air"
        )
        result = hfss.modeler.subtract(
            blank_list=air,
            tool_list=conductor,
            keep_originals=True,
        )
        if result is False:
            raise RuntimeError("Failed subtracting conductor from AirInterior")
        air.solve_inside = True

        # Match the openEMS lumped-port spans: from the top of the thin strip
        # to the inner surface of the top wall. Sheets lie at x=7.0 and 58.3 mm.
        z_strip = H_THIN / 2
        z_ground = TOTAL_HEIGHT / 2 - WALL
        y0, y1 = -CENTER_WIDTH/2, CENTER_WIDTH/2

        for name, xp in (("Port1", PORT1_X), ("Port2", PORT2_X)):
            sheet = need(
                hfss.modeler.create_rectangle(
                    orientation="YZ",
                    origin=[xp, y0, z_strip],
                    sizes=[y1-y0, z_ground-z_strip],
                    name=f"{name}_Sheet",
                ),
                f"{name}_Sheet",
            )
            # Integration line runs vertically from centre strip toward top wall.
            port = hfss.lumped_port(
                assignment=sheet.name,
                name=name,
                impedance=Z0,
                renormalize=True,
                integration_line=[
                    [xp, 0, z_strip],
                    [xp, 0, z_ground],
                ],
            )
            need(port, name)

        setup = need(hfss.create_setup("Setup1"), "Setup1")
        setup.props["Frequency"] = F_ADAPT
        setup.props["MaximumPasses"] = MAX_PASSES
        setup.props["MinimumPasses"] = 2
        setup.props["MinimumConvergedPasses"] = 1
        setup.props["DeltaS"] = DELTA_S
        need(setup.update(), "Setup1 update")
        sweep = need(
            setup.create_frequency_sweep(
                unit="GHz",
                name="Sweep1",
                start_frequency=F_START_GHZ,
                stop_frequency=F_STOP_GHZ,
                sweep_type="Interpolating",
                save_fields=False,
            ),
            "Sweep1",
        )

        manifest = {
            "status": "RUN13 GEOMETRY RECONSTRUCTION - VALIDATION REQUIRED",
            "project": str(PROJECT),
            "units": "mm",
            "enclosure_mm": [TOTAL_LENGTH, TOTAL_WIDTH, TOTAL_HEIGHT],
            "wall_mm": WALL,
            "centre_strip_width_mm": CENTER_WIDTH,
            "launch_lengths_mm": [INPUT_LAUNCH, OUTPUT_LAUNCH],
            "thin_height_mm": H_THIN,
            "thick_height_mm": H_THICK,
            "section_lengths_mm": SECTION_LENGTHS,
            "port_planes_x_mm": [PORT1_X, PORT2_X],
            "port_impedance_ohm": Z0,
            "frequency_sweep_GHz": [F_START_GHZ, F_STOP_GHZ],
            "caveat": (
                "HFSS lumped ports are not numerically identical to openEMS "
                "AddLumpedPort. Verify port integration, S-parameter reference "
                "impedance, mesh convergence, and compare curves before use."
            ),
            "sections": section_records,
        }
        (OUT / "run13_geometry_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        hfss.save_project()
        print(f"\nSaved new reference project: {PROJECT}")
        print(f"Geometry manifest: {OUT / 'run13_geometry_manifest.json'}")
        print("Geometry created. Starting first HFSS solve...")
        solved = hfss.analyze_setup("Setup1")
        print(f"analyze_setup returned: {solved}")
        hfss.save_project()

        if solved:
            # Export named S-parameter CSV expressions from HFSS directly.
            try:
                report = hfss.post.create_report(
                    expressions=["dB(S(Port1,Port1))", "dB(S(Port2,Port1))"],
                    setup_sweep_name="Setup1 : Sweep1",
                    primary_sweep_variable="Freq",
                    report_category="Standard",
                    plot_type="Rectangular Plot",
                )
                if report:
                    hfss.post.export_report_to_csv(
                        plot_name,
                        output_file_name=str(output_csv)
                    )
                    print(f"Requested S-parameter CSV export: {OUT / 'Run13_HFSS_Sparams.csv'}")
            except Exception as exc:
                print(f"Solver completed, but report export needs attention: {exc}")
        else:
            print("HFSS did not report a successful solve. Inspect AEDT Message Manager.")
        input("Press Enter to close AEDT (project is saved).")

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
