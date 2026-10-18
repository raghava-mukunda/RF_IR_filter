from ansys.aedt.core import Desktop, Hfss

desktop = Desktop(
    student_version=True,
    non_graphical=False,
    new_desktop=True,
)

hfss = Hfss(
    project="RF_Filter_Material_Check",
    design="Material_Check",
    solution_type="Modal",
    version="2025.2",
    new_desktop=False,
)

print("Vacuum exists:", "vacuum" in hfss.materials.material_keys)

region = hfss.modeler.create_region(
    pad_value=0,
    pad_type="Absolute Offset",
    name="RegionCheck",
)

print("Region created:", region.name if region else None)
print("Region material:", region.material_name if region else None)
print("Region solve_inside:", region.solve_inside if region else None)

input("Inspection complete. Press Enter to close.")
desktop.release_desktop()