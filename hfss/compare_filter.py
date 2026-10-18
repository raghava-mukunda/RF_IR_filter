
from pathlib import Path
import json
from datetime import datetime
from ansys.aedt.core import Desktop, Hfss

# ============================================================
# Inspect the actual solved HFSS project
# ============================================================

PROJECT_FILE = Path(
    r"C:\Users\Admin\Documents\Ansoft\RF_Filter_Run013.aedt"
)
DESIGN_NAME = "LPF_Run013"

OUT = Path("hfss") / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

if not PROJECT_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find HFSS project: {PROJECT_FILE}"
    )

desktop = Desktop(
    student_version=True,
    non_graphical=False,
    new_desktop=False,
)

try:
    # IMPORTANT: pass the full .aedt path, not just the project name.
    hfss = Hfss(
        project=str(PROJECT_FILE),
        design=DESIGN_NAME,
        version="2025.2",
        new_desktop=False,
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_file": str(PROJECT_FILE),
        "design": DESIGN_NAME,
        "objects": [],
        "boundaries": [],
        "setups": [],
        "warnings": [],
    }

    print("\n========== LOADED PROJECT ==========")
    print("Project:", hfss.project_name)
    print("Design:", hfss.design_name)
    print("Units:", hfss.modeler.model_units)

    print("\n========== OBJECTS ==========")
    for name, obj in hfss.modeler.objects_by_name.items():
        item = {
            "name": name,
            "material": obj.material_name,
            "solve_inside": obj.solve_inside,
        }
        report["objects"].append(item)
        print(item)

    print("\n========== BOUNDARIES ==========")
    for boundary in hfss.boundaries:
        item = {
            "name": boundary.name,
            "type": boundary.type,
        }
        try:
            item["properties"] = dict(boundary.props)
        except Exception:
            item["properties"] = {}
        report["boundaries"].append(item)
        print(json.dumps(item, indent=2, default=str))

    print("\n========== SETUPS ==========")
    for setup in hfss.setups:
        item = {
            "name": setup.name,
            "properties": dict(setup.props),
        }
        report["setups"].append(item)
        print(json.dumps(item, indent=2, default=str))

    # Confirm the design was loaded rather than silently inspecting
    # a newly created, empty design.
    if not report["objects"]:
        raise RuntimeError(
            "The loaded design has no geometry objects. "
            "Stop here: do not interpret this as a filter result."
        )

    if not report["boundaries"]:
        raise RuntimeError(
            "No boundaries were found in the loaded design."
        )

    # The solved setup is expected to be Setup1 with Sweep1.
    setup_names = [s.name for s in hfss.setups]
    if "Setup1" not in setup_names:
        report["warnings"].append(
            f"Setup1 not found. Available setups: {setup_names}"
        )

    # Record the design's available solution data.
    print("\n========== SOLUTION DATA ==========")
    try:
        print("Available variations:", hfss.available_variations)
    except Exception as exc:
        report["warnings"].append(
            f"Could not read available variations: {exc}"
        )
        print("Could not read variations:", exc)

    json_path = OUT / "hfss_diagnostic.json"
    json_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print("\nDiagnostic saved:", json_path.resolve())
    print("Object count:", len(report["objects"]))
    print("Boundary count:", len(report["boundaries"]))
    print("Setup count:", len(report["setups"]))

finally:
    desktop.release_desktop(close_projects=False)
