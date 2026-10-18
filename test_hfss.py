
from ansys.aedt.core import Desktop

desktop = Desktop(
    student_version=True,
    non_graphical=False,
    new_desktop=True,
)

print("Connected to Ansys Electronics Desktop.")
print("AEDT version:", desktop.aedt_version)
print("Installed versions:", desktop.installed_versions)

input("Press Enter to finish and close this script...")

# Release the PyAEDT connection without explicitly closing AEDT.
desktop.release_desktop(close_projects=False)
