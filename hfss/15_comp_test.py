"""
Build and solve an analytically synthesized 10 GHz low-pass filter reference
inside Ansys Electronics Desktop Circuit using PyAEDT.

Architecture:
  15th-order, 0.1 dB-ripple Chebyshev ladder
  Equal 50-ohm input/output references
  Series L at odd-numbered stages, shunt C at even-numbered stages
  Passband edge selected so |S21| = -3 dB at 10 GHz
  Ideal prototype is calculated to give about -62.1 dB at 12 GHz

This is an AEDT CIRCUIT schematic (ideal lumped elements), NOT a 3D EM geometry.
It provides the correct electrical target against which a later physical HFSS
model can be compared. Parasitics and distributed spurious passbands are not
included in this ideal circuit.

Run from the repository root:
  .venv\\Scripts\\python.exe build_10GHz_LPF_in_AEDT_Circuit.py
"""
from pathlib import Path
import math
import json
import traceback

from ansys.aedt.core import Desktop, Circuit

ROOT = Path.cwd()
OUT = ROOT / "hfss" / "chebyshev_lpf_10GHz" / "circuit_reference"
PROJECT_NAME = "Chebyshev15_10GHz_Circuit"
PROJECT_PATH = OUT / f"{PROJECT_NAME}.aedt"
DESIGN_NAME = "Chebyshev15_LPF"
AEDT_VERSION = "2025.2"
STUDENT_VERSION = True

ORDER = 15
RIPPLE_DB = 0.1
Z0 = 50.0
F_MINUS3DB = 10e9
F_STOP_CHECK = 12e9
F_MAX = 30e9


def chebyshev_g(n, ripple_db):
    beta = math.log(1.0 / math.tanh(ripple_db / 17.37))
    gamma = math.sinh(beta / (2.0 * n))
    a = [math.sin((2*k - 1) * math.pi / (2*n)) for k in range(1, n+1)]
    b = [gamma**2 + math.sin(k * math.pi / n)**2 for k in range(1, n+1)]
    g = [0.0] * (n + 1)
    g[0] = 2.0 * a[0] / gamma
    for k in range(1, n):
        g[k] = 4.0 * a[k-1] * a[k] / (b[k-1] * g[k-1])
    g[n] = 1.0 if n % 2 else (1.0 / math.tanh(beta / 4.0))**2
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if PROJECT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing project: {PROJECT_PATH}\n"
            "Change PROJECT_NAME in this script to create another iteration."
        )

    eps = math.sqrt(10**(RIPPLE_DB/10.0) - 1.0)
    t3 = math.sqrt((10**(3.0/10.0) - 1.0) / eps**2)
    omega3 = math.cosh(math.acosh(t3) / ORDER)
    fp = F_MINUS3DB / omega3
    g = chebyshev_g(ORDER, RIPPLE_DB)

    elements = []
    for index, gi in enumerate(g[:ORDER], start=1):
        if index % 2:  # series L
            val = Z0 * gi / (2.0 * math.pi * fp)
            elements.append({"index": index, "type": "L", "g": gi, "value_si": val})
        else:  # shunt C
            val = gi / (Z0 * 2.0 * math.pi * fp)
            elements.append({"index": index, "type": "C", "g": gi, "value_si": val})

    (OUT / "synthesis_values.json").write_text(
        json.dumps({
            "order": ORDER,
            "ripple_db": RIPPLE_DB,
            "reference_impedance_ohm": Z0,
            "passband_edge_GHz": fp / 1e9,
            "target_minus3db_GHz": F_MINUS3DB / 1e9,
            "element_values": elements,
            "load_g": g[ORDER],
            "notes": [
                "Ideal LC circuit model, not 3D EM.",
                "Physical HFSS implementation must be validated separately."
            ]
        }, indent=2), encoding="utf-8"
    )

    desktop = None
    try:
        desktop = Desktop(
            version=AEDT_VERSION,
            student_version=STUDENT_VERSION,
            non_graphical=False,
            new_desktop=True
        )
        circuit = Circuit(
            project=str(PROJECT_PATH),
            design=DESIGN_NAME,
            version=AEDT_VERSION,
            new_desktop=False
        )
        sch = circuit.modeler.schematic
        sch.schematic_units = "mil"

        # Input and output circuit ports. The circuit design's S-parameter
        # analysis uses the standard 50-ohm reference for these interface ports.
        p_in = sch.create_interface_port(name="Port1", location=[0, 0])
        p_out = sch.create_interface_port(name="Port2", location=[(ORDER + 2) * 600, 0])

        # Place series elements on the main signal row, and shunt capacitors
        # below it. Connectivity is explicitly defined pin-to-pin.
        current_pin = p_in.pins[0]
        last_node_pin = None
        ground_count = 0
        component_x = 600

        for element in elements:
            idx = element["index"]
            value = element["value_si"]
            x = idx * component_x
            if element["type"] == "L":
                comp = sch.create_inductor(
                    name=f"L{idx}", value=value, location=[x, 0], angle=0
                )
                current_pin.connect_to_component(
                    assignment=comp.pins[0], use_wire=True
                )
                current_pin = comp.pins[1]
                last_node_pin = current_pin
            else:
                cap = sch.create_capacitor(
                    name=f"C{idx}", value=value, location=[x, -350], angle=90
                )
                # Capacitor's pin 0 joins the series signal node; pin 1 goes to ground.
                current_pin.connect_to_component(
                    assignment=cap.pins[0], use_wire=True
                )
                gnd = sch.create_gnd(location=[x, -850])
                cap.pins[1].connect_to_component(
                    assignment=gnd.pins[0], use_wire=True
                )
                current_pin = cap.pins[0]
                last_node_pin = current_pin

        current_pin.connect_to_component(
            assignment=p_out.pins[0], use_wire=True
        )

        setup = circuit.create_setup("FilterSweep")
        setup.props["SweepDefinition"]["Data"] = "LINC 0.1GHz 30GHz 2990"
        setup.update()
        circuit.save_project()

        print(f"Project: {PROJECT_PATH}")
        print(f"Calculated passband edge: {fp/1e9:.6f} GHz")
        print(f"Target |S21| at 10 GHz: -3 dB")
        print("Running AEDT Circuit analysis...")
        ok = circuit.analyze_setup("FilterSweep")
        print(f"analyze_setup returned: {ok}")
        circuit.save_project()
        if not ok:
            raise RuntimeError("Circuit analysis failed. Check AEDT Message Manager.")

        s2p_path = OUT / "Chebyshev15_10GHz_AEDT_Circuit.s2p"
        exported = circuit.export_touchstone(
            setup="FilterSweep",
            output_file=str(s2p_path),
            renormalization=True,
            impedance=Z0
        )
        print(f"Touchstone export return: {exported}")
        print(f"Touchstone file exists: {s2p_path.exists()}")
        input("Press Enter to close AEDT.")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if desktop:
            try:
                desktop.release_desktop(close_projects=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()
