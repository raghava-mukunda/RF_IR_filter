"""
Repair the two internal wave ports in the existing candidate project.

The latest AEDT messages report that Port1 and Port2 are internal to the
solution region and are not backed by PEC objects. This script recreates the
ports with create_pec_cap=True, saves, and retries Setup1.

This is a solver-diagnostic step, NOT CliQ10 validation. The hollow absorber
channels are still absent from the current candidate.

Run from repository root:
    .\\.venv\\Scripts\\python.exe hfss\\repair_ports_and_solve.py
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
PORTS = [
    ("Port1", "PortSheet1"),
    ("Port2", "PortSheet2"),
]
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

        # Remove old port boundaries before re-creating them.
        current_boundaries = {
            getattr(boundary, "name", ""): boundary
            for boundary in hfss.boundaries
        }
        for port_name, _sheet_name in PORTS:
            if port_name in current_boundaries:
                # In PyAEDT 1.9.0 boundary deletion is handled by the
                # boundary manager, not Hfss.delete_boundary().
                deleted = hfss.oboundary.delete(port_name)
                print(f"Deleted old {port_name}: {deleted}")

        # Ensure the PEC cap is explicitly created for each internal port.
        for port_name, sheet_name in PORTS:
            if sheet_name not in hfss.modeler.object_names:
                raise RuntimeError(
                    f"Port sheet '{sheet_name}' not found. "
                    "Existing project geometry differs from expected."
                )
            port = hfss.wave_port(
                assignment=sheet_name,
                modes=2,
                impedance=50,
                renormalize=True,
                name=port_name,
                create_pec_cap=True,
                port_on_plane=True,
            )
            if port is None or port is False:
                raise RuntimeError(f"Failed to recreate {port_name}.")
            print(f"Recreated {port_name} on {sheet_name} with PEC cap.")

        hfss.save_project()
        print("Saved project with repaired port definitions.")
        print("Retrying Setup1...")
        result = hfss.analyze_setup(SETUP_NAME)
        print(f"analyze_setup('{SETUP_NAME}') returned: {result}")
        hfss.save_project()

        if result:
            print(
                "HFSS reports Setup1 completed. This is only a numerical solve "
                "of the current simplified geometry, NOT CliQ10 validation."
            )
        else:
            print(
                "Setup1 still failed. Please send the new AEDT Message Manager "
                "errors; do not change unrelated settings yet."
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
