"""
Export S-parameters from the already-solved CliQ10 Iteration 01 project.
Does NOT rebuild geometry or rerun the solver.

Run from repository root:
    .\\.venv\\Scripts\\python.exe hfss\\export_cliq10_iter01_sparams.py
"""

from pathlib import Path
import traceback
from ansys.aedt.core import Desktop, Hfss

PROJECT = Path(
    r"C:\Users\Admin\Desktop\IR_filter\RF_IR_filter\hfss\cliq10_iter01"
    r"\CliQ10_Iter01.aedt"
)
DESIGN = "CliQ10_Iter01"
AEDT_VERSION = "2025.2"
SETUP = "Setup1"
SWEEP = "Sweep1"


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

        out_dir = PROJECT.parent
        touchstone = (out_dir / "CliQ10_Iter01.s2p").resolve()
        csv_path = (out_dir / "CliQ10_Iter01_Sparams.csv").resolve()

        print("Project opened. No geometry changes will be made.")
        print(f"Exporting solved network data to: {touchstone}")

        result = hfss.export_touchstone(
            setup=SETUP,
            sweep=SWEEP,
            output_file=str(touchstone),
            renormalization=False,
        )
        print(f"export_touchstone returned: {result}")

        if not touchstone.exists():
            print(
                "Touchstone file was not found at the requested path. "
                "Check the returned value and AEDT messages."
            )
        else:
            print(f"Touchstone created: {touchstone} ({touchstone.stat().st_size} bytes)")

        # Also try retrieving solved sweep data directly and exporting it.
        try:
            solution = hfss.post.get_solution_data(
                expressions=[
                    "dB(S(Port1,Port1))",
                    "dB(S(Port2,Port1))",
                ],
                setup_sweep_name=f"{SETUP} : {SWEEP}",
            )
            if solution:
                solution.export_data_to_csv(str(csv_path))
                print(f"Solution-data CSV exported: {csv_path}")
            else:
                print(
                    "Direct solution-data retrieval returned no data; "
                    "the Touchstone export is the primary output."
                )
        except Exception as exc:
            print(f"Direct CSV export unavailable: {exc}")

        print("\nExport attempt complete. Solver was not rerun.")
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
