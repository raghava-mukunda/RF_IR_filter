"""
Rebuild the existing CliQ10-inspired candidate's wave-port sheets so they fit
inside the enclosure aperture rather than spanning through the copper walls.

This addresses the specific AEDT intersections reported between PortSheet3/4
(PEC caps created for the wave ports) and the four enclosure walls.

IMPORTANT:
- This only repairs the port geometry of the current simplified candidate.
- The model still lacks the hollow absorber channels and is NOT a validated
  CliQ10 design.
- The script edits the existing CliQ10_Inspired_Candidate.aedt project.
- It does not touch Run 013.

Run from repository root:
    .\.venv\Scripts\python.exe hfss\rebuild_ports_inside_aperture.py
"""

from pathlib import Path
import traceback

from ansys.aedt.core import Desktop, Hfss

PROJECT = Path(
    r"C:\Users\Admin\Desktop\IR_filter\RF_IR_filter\hfss\cliq10_candidate"
    r"\CliQ10_Inspired_Candidate.aedt"
)
DESIGN = "CliQ10_Candidate"
AEDT_VERSION = "2025.2"
LENGTH = 65.3
WIDTH = 5.0
HEIGHT = 4.6
WALL_T = 0.25
PORT_MODES = 2
SETUP_NAME = "Setup1"


def main():
    desktop = None
    hfss = None
    try:
        if not PROJECT.exists():
            raise FileNotFoundError(f"Project not found: {PROJECT}")

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

        # Delete existing wave-port boundaries first.
        boundary_names = {
            getattr(boundary, "name", "") for boundary in hfss.boundaries
        }
        for port_name in ("Port1", "Port2"):
            if port_name in boundary_names:
                hfss.oboundary.delete(port_name)
                print(f"Deleted boundary {port_name}")

        # Delete old port sheets and any caps created by create_pec_cap=True.
        # These names are grounded in the preceding run's AEDT messages.
        names_now = set(hfss.modeler.object_names)
        for object_name in ("PortSheet1", "PortSheet2", "PortSheet3", "PortSheet4"):
            if object_name in names_now:
                deleted = hfss.modeler.delete(object_name)
                print(f"Deleted old port geometry {object_name}: {deleted}")

        # Make the sheet fit strictly within the open cross-section between
        # the four 0.25 mm walls. It touches the centre conductor at the port
        # plane but does not pass through the enclosure walls.
        inner_w = WIDTH - 2 * WALL_T
        inner_h = HEIGHT - 2 * WALL_T
        y0 = -WIDTH / 2 + WALL_T
        z0 = -HEIGHT / 2 + WALL_T

        for port_name, x_pos in (("Port1", 0.0), ("Port2", LENGTH)):
            sheet_name = "PortSheet1" if port_name == "Port1" else "PortSheet2"
            sheet = hfss.modeler.create_rectangle(
                orientation="YZ",
                origin=[x_pos, y0, z0],
                sizes=[inner_w, inner_h],
                name=sheet_name,
            )
            if sheet is None or sheet is False:
                raise RuntimeError(f"Could not create {sheet_name}")

            port = hfss.wave_port(
                assignment=sheet_name,
                modes=PORT_MODES,
                impedance=50,
                renormalize=True,
                name=port_name,
                create_pec_cap=True,
                port_on_plane=True,
            )
            if port is None or port is False:
                raise RuntimeError(f"Could not create {port_name}")
            print(
                f"Created {port_name}: sheet {inner_w:.3f} x {inner_h:.3f} mm "
                f"at x={x_pos:.3f} mm, with PEC cap."
            )

        hfss.save_project()
        print("\nPort sheets rebuilt and project saved.")
        print("Retrying Setup1...")
        result = hfss.analyze_setup(SETUP_NAME)
        print(f"analyze_setup('{SETUP_NAME}') returned: {result}")
        hfss.save_project()

        if result:
            print(
                "HFSS reports Setup1 completed. This is only a solve of the "
                "simplified candidate; it does not validate CliQ10 performance."
            )
        else:
            print(
                "Setup1 still failed. Send the complete new AEDT Message Manager "
                "messages and PowerShell output."
            )
        input("Press Enter to exit.")
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
