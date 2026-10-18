
from ansys.aedt.core import Desktop, Hfss

# ============================================================
# RF LOW-PASS FILTER — OPENEMS RUN 013 RECONSTRUCTION
# Ansys Electronics Desktop Student 2025 R2
# PyAEDT 1.9.0
# All geometry dimensions are in mm.
# ============================================================

PROJECT_NAME = "RF_Filter_Run013"
DESIGN_NAME = "LPF_Run013"

# ------------------------------------------------------------
# 1. Start AEDT and create the HFSS design
# ------------------------------------------------------------

desktop = Desktop(
    student_version=True,
    non_graphical=False,
    new_desktop=True,
)

hfss = Hfss(
    project=PROJECT_NAME,
    design=DESIGN_NAME,
    solution_type="Modal",
    version="2025.2",
    new_desktop=False,
)

hfss.modeler.model_units = "mm"

# ------------------------------------------------------------
# 2. Copper enclosure
#
# Outer dimensions:
#   x: 0 to 65.3 mm
#   y: -2.5 to +2.5 mm
#   z: -2.3 to +2.3 mm
#
# Top/bottom walls stop at the inner faces of side walls.
# ------------------------------------------------------------

walls = [
    (
        "BottomWall",
        [0, -2.3, -2.3],
        [65.3, 4.6, 0.2],
    ),
    (
        "TopWall",
        [0, -2.3, 2.1],
        [65.3, 4.6, 0.2],
    ),
    (
        "SideWallNegY",
        [0, -2.5, -2.3],
        [65.3, 0.2, 4.6],
    ),
    (
        "SideWallPosY",
        [0, 2.3, -2.3],
        [65.3, 0.2, 4.6],
    ),
]

for name, origin, sizes in walls:
    obj = hfss.modeler.create_box(
        origin=origin,
        sizes=sizes,
        name=name,
        material="copper",
    )
    if not obj:
        raise RuntimeError(f"Failed to create {name}")

# ------------------------------------------------------------
# 3. Stepped central copper conductor
#
# Adjacent sections touch at their x boundaries.
# Do not introduce gaps between sections.
# ------------------------------------------------------------

sections = [
    (0.0, 8.0, 0.143359638314843),
    (8.0, 12.0736569145949, 0.143359638314843),
    (12.0736569145949, 19.6060435216811, 0.381745831361027),
    (19.6060435216811, 27.6770556020086, 0.143359638314843),
    (27.6770556020086, 37.6229443979914, 0.381745831361027),
    (37.6229443979914, 45.6939564783189, 0.143359638314843),
    (45.6939564783189, 53.2263430854051, 0.381745831361027),
    (53.2263430854051, 57.3, 0.143359638314843),
    (57.3, 65.3, 0.143359638314843),
]

conductors = []

for i, (x1, x2, half_height) in enumerate(sections, start=1):
    name = f"Conductor_{i:02d}"

    obj = hfss.modeler.create_box(
        origin=[x1, -0.6, -half_height],
        sizes=[x2 - x1, 1.2, 2 * half_height],
        name=name,
        material="copper",
    )

    if not obj:
        raise RuntimeError(f"Failed to create {name}")

    conductors.append(obj)

# ------------------------------------------------------------
# 4. Internal air volume
#
# Start with the enclosure interior, then subtract the central
# copper conductor sections from the air solid.
#
# The resulting air body does not overlap the conductor.
# ------------------------------------------------------------

air = hfss.modeler.create_box(
    origin=[0, -2.3, -2.1],
    sizes=[65.3, 4.6, 4.2],
    name="AirRegion",
    material="air",
)

if not air:
    raise RuntimeError("Failed to create AirRegion")

# Preserve the copper tool solids while cutting their shapes
# out of the air volume.
subtract_ok = hfss.modeler.subtract(
    blank_list=air,
    tool_list=conductors,
    keep_originals=True,
)

if not subtract_ok:
    raise RuntimeError(
        "Subtracting the conductor from AirRegion failed. "
        "Inspect the AEDT model before proceeding."
    )

# Explicitly enable field solving in the air volume.
air.solve_inside = True

# ------------------------------------------------------------
# 5. Wave-port sheets at x = 0 and x = 65.3 mm
#
# Aperture dimensions:
#   y: -2.3 to +2.3 mm
#   z: -2.1 to +2.1 mm
#
# Two modes are requested at each port.
# ------------------------------------------------------------

port1 = hfss.modeler.create_rectangle(
    orientation="YZ",
    origin=[0, -2.3, -2.1],
    sizes=[4.6, 4.2],
    name="Port1Sheet",
)

port2 = hfss.modeler.create_rectangle(
    orientation="YZ",
    origin=[65.3, -2.3, -2.1],
    sizes=[4.6, 4.2],
    name="Port2Sheet",
)

if not port1 or not port2:
    raise RuntimeError("Failed to create one or both port sheets")

p1 = hfss.wave_port(
    assignment=port1.name,
    name="Port1",
    modes=2,
    impedance=50,
    renormalize=True,
)

p2 = hfss.wave_port(
    assignment=port2.name,
    name="Port2",
    modes=2,
    impedance=50,
    renormalize=True,
)

if not p1 or not p2:
    raise RuntimeError("Failed to assign one or both wave ports")

# ------------------------------------------------------------
# 6. Adaptive solution setup
# ------------------------------------------------------------

setup = hfss.create_setup("Setup1")

setup.props["Frequency"] = "40GHz"
setup.props["MaximumPasses"] = 10
setup.props["MinimumPasses"] = 2
setup.props["MinimumConvergedPasses"] = 1
setup.props["DeltaS"] = 0.02

if not setup.update():
    raise RuntimeError("Failed to update Setup1")

# ------------------------------------------------------------
# 7. Frequency sweep: 0.01 to 40 GHz
# ------------------------------------------------------------

sweep = setup.create_frequency_sweep(
    unit="GHz",
    name="Sweep1",
    start_frequency=0.01,
    stop_frequency=40.0,
    sweep_type="Interpolating",
    save_fields=False,
)

if not sweep:
    raise RuntimeError("Failed to create Sweep1")

# ------------------------------------------------------------
# 8. Diagnostic report
# ------------------------------------------------------------

print("\n========== MODEL DIAGNOSTICS ==========")
print("Project:", PROJECT_NAME)
print("Design:", DESIGN_NAME)
print("Units:", hfss.modeler.model_units)

print("\nObjects:")
for name, obj in hfss.modeler.objects_by_name.items():
    try:
        print(
            f"{name}: "
            f"material={obj.material_name}, "
            f"solve_inside={obj.solve_inside}"
        )
    except Exception as exc:
        print(f"{name}: could not inspect properties: {exc}")

print("\nBoundaries:")
for boundary in hfss.boundaries:
    print(f"{boundary.name}: {boundary.type}")

print("\nAdaptive frequency: 40 GHz")
print("Maximum passes: 10")
print("Delta S: 0.02")
print("Sweep: 0.01–40 GHz")

# Save before starting the solver.
if not hfss.save_project():
    raise RuntimeError("Failed to save the HFSS project")

print("\nProject saved.")
input("Review the model in AEDT, then press Enter to start solving...")

# ------------------------------------------------------------
# 9. Solve
# ------------------------------------------------------------

try:
    success = hfss.analyze("Setup1")
    print("\nAnalysis completed:", success)

    if success:
        print("HFSS reported that the analysis completed.")
        print(
            "Next: inspect S(Port2,Port1) and compare it with "
            "the openEMS Run 013 reference."
        )
    else:
        print(
            "HFSS analysis failed. Inspect the AEDT Message Manager "
            "for the first detailed solver error."
        )

    hfss.save_project()

    input("Press Enter to close the AEDT session...")

finally:
    desktop.release_desktop(close_projects=False)
