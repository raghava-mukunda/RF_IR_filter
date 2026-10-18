import inspect
from ansys.aedt.core import Desktop, Hfss

desktop = Desktop(
    student_version=True,
    non_graphical=False,
    new_desktop=True,
)

hfss = Hfss(
    project="RF_Filter_API_Check",
    design="API_Check",
    solution_type="Modal",
    version="2025.2",
    new_desktop=False,
)

print("create_box:", inspect.signature(hfss.modeler.create_box))
print("wave_port:", inspect.signature(hfss.wave_port))
print("create_region:", inspect.signature(hfss.modeler.create_region))

print("Model units:", hfss.modeler.model_units)
print("Material names:", list(hfss.materials.material_keys)[:20])

input("Inspection complete. Press Enter to close.")
desktop.release_desktop()