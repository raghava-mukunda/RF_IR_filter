"""
Add an air region and run the existing CliQ10-inspired HFSS candidate.

This is a diagnostic solver script, not a validated CliQ10 design.
It opens the existing project created by build_herd_filter_recovery.py,
adds an air region with Solve Inside enabled, saves, then attempts Setup1.

Run from repo root:
    .\.venv\Scripts\python.exe hfss\add_air_region_and_solve.py
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
SETUP_NAME = "Setup1"
REGION_NAME = "AirRegion_SolveInside"

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

        # If this script has already been run, avoid creating duplicate regions.
        existing = {str(name).lower() for name in hfss.modeler.object_names}
        if REGION_NAME.lower() not in existing:
            region = hfss.modeler.create_region(
                pad_value=20,
                pad_type="Percentage Offset",
                name=REGION_NAME,
            )
            if region is None or region is False:
                raise RuntimeError("AEDT failed to create the air region.")
            try:
                region.material_name = "air"
            except Exception:
                pass
            try:
                region.solve_inside = True
            except Exception:
                # create_region normally creates a dielectric air region with
                # Solve Inside enabled. We still verify the resulting object.
                pass

        hfss.save_project()

        # Print object/material state for diagnosis.
        print("\n--- Objects and material/solve-inside state ---")
        for name in hfss.modeler.object_names:
            try:
                obj = hfss.modeler[name]
                material = getattr(obj, "material_name", "unknown")
                solve_inside = getattr(obj, "solve_inside", "unknown")
                print(f"{name}: material={material}, solve_inside={solve_inside}")
            except Exception as exc:
                print(f"{name}: could not inspect ({exc})")

        # Explicitly verify the region exists.
        if REGION_NAME.lower() not in {str(n).lower() for n in hfss.modeler.object_names}:
            raise RuntimeError(
                f"{REGION_NAME} was not found after create_region; not solving."
            )

        print(f"\nAir region added/confirmed: {REGION_NAME}")
        print("Saved project. Starting Setup1...")
        result = hfss.analyze_setup(SETUP_NAME)
        print(f"analyze_setup('{SETUP_NAME}') returned: {result}")
        hfss.save_project()

        if not result:
            print(
                "\nHFSS did not report a successful solve. Read the AEDT Message "
                "Manager for the first red error; this script will not claim success."
            )
        else:
            print(
                "\nHFSS reports the setup completed. This only validates the current "
                "simplified geometry numerically; hollow absorber channels are still "
                "missing, so it is NOT a validated CliQ10 reproduction."
            )

        input("\nPress Enter to exit this script.")
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
