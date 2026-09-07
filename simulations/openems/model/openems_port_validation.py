import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OPENEMS_ROOT = os.environ.get("OPENEMS_INSTALL_PATH")
if OPENEMS_ROOT:
    os.add_dll_directory(OPENEMS_ROOT)

from CSXCAD import ContinuousStructure
from openEMS import openEMS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIM_PATH = PROJECT_ROOT / "simulations" / "openems" / "results" / "port_validation"
RESULTS_PATH = PROJECT_ROOT / "results" / "em"
SIM_PATH.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

C0 = 299792458.0
unit = 1e-3

f0 = 10e9
fc = 10e9
f_start = 1e9
f_stop = 15e9

# Small sanity model: ONLY validate the microstrip ports.
line_length = 20.0
line_width = 3.5577
substrate_width = 30.0
substrate_length = 30.0
substrate_thickness = 1.524
copper_thickness = 0.035
substrate_epsr = 3.38

port1_x = -7.0
port2_x = +7.0
port_y = 0.0
feed_R = 50.0

SimBox = np.array([40.0, 30.0, 20.0])
lambda_min_mm = (C0 / (f0 + fc)) / unit
coarse_step = lambda_min_mm / 15.0
substrate_step = substrate_thickness / 6.0

FDTD = openEMS(NrTS=200000, EndCriteria=1e-5)
FDTD.SetGaussExcite(f0, fc)
FDTD.SetBoundaryCond(["MUR"] * 6)

CSX = ContinuousStructure()
FDTD.SetCSX(CSX)
mesh = CSX.GetGrid()
mesh.SetDeltaUnit(unit)

substrate = CSX.AddMaterial("substrate", epsilon=substrate_epsr)
substrate.AddBox(
    start=[-substrate_width/2, -substrate_length/2, 0.0],
    stop=[substrate_width/2, substrate_length/2, substrate_thickness],
)

ground = CSX.AddMetal("ground")
ground.AddBox(
    start=[-substrate_width/2, -substrate_length/2, -copper_thickness],
    stop=[substrate_width/2, substrate_length/2, 0.0],
)

signal = CSX.AddMetal("signal")
signal.AddBox(
    start=[-line_length/2, -line_width/2, substrate_thickness],
    stop=[line_length/2, line_width/2, substrate_thickness + copper_thickness],
)

# EXACT construction used by the previously working 2-port model.
port1 = FDTD.AddLumpedPort(
    port_nr=1,
    R=feed_R,
    start=[port1_x, port_y, -copper_thickness],
    stop=[port1_x, port_y, substrate_thickness + copper_thickness],
    p_dir="z",
    excite=1.0,
)

port2 = FDTD.AddLumpedPort(
    port_nr=2,
    R=feed_R,
    start=[port2_x, port_y, -copper_thickness],
    stop=[port2_x, port_y, substrate_thickness + copper_thickness],
    p_dir="z",
    excite=0.0,
)

x_uniform = np.arange(
    -SimBox[0]/2, SimBox[0]/2 + coarse_step*0.5, coarse_step
)
x_special = np.array([
    -SimBox[0]/2, -substrate_width/2, -line_length/2,
    port1_x, port2_x, line_length/2,
    substrate_width/2, SimBox[0]/2
])
x_lines = np.unique(np.round(np.concatenate([x_uniform, x_special]), 12))

y_uniform = np.arange(
    -SimBox[1]/2, SimBox[1]/2 + coarse_step*0.5, coarse_step
)
y_special = np.array([
    -SimBox[1]/2, -line_width/2, 0.0, line_width/2, SimBox[1]/2
])
y_lines = np.unique(np.round(np.concatenate([y_uniform, y_special]), 12))

z_lower = np.arange(-SimBox[2]/2, -copper_thickness, coarse_step)
z_upper = np.arange(
    substrate_thickness + copper_thickness + coarse_step,
    SimBox[2]/2 + coarse_step*0.5,
    coarse_step,
)
z_lines = np.unique(np.round(np.concatenate([
    z_lower,
    [
        -copper_thickness,
        0.0,
        substrate_step,
        2*substrate_step,
        3*substrate_step,
        4*substrate_step,
        5*substrate_step,
        substrate_thickness,
        substrate_thickness + copper_thickness,
    ],
    z_upper,
]), 12))

mesh.SetLines("x", x_lines.tolist())
mesh.SetLines("y", y_lines.tolist())
mesh.SetLines("z", z_lines.tolist())

print("=" * 72)
print("openEMS MICROSTRIP PORT VALIDATION")
print("=" * 72)
print(f"Simulation directory : {SIM_PATH}")
print(f"Line width           : {line_width:.4f} mm")
print(f"Port 1 x             : {port1_x:.3f} mm")
print(f"Port 2 x             : {port2_x:.3f} mm")
print(f"Mesh X/Y/Z           : {len(x_lines)} / {len(y_lines)} / {len(z_lines)}")
print(f"Maximum mesh step    : {coarse_step:.4f} mm")
print("=" * 72)

xml_path = SIM_PATH / "port_validation.xml"
if xml_path.exists():
    xml_path.unlink()
CSX.Write2XML(str(xml_path))
print(f"XML written          : {xml_path}")
print("Starting openEMS...")
print()

FDTD.Run(str(SIM_PATH), cleanup=True)

freq = np.linspace(f_start, f_stop, 401)

port1.CalcPort(str(SIM_PATH), freq, ref_impedance=50.0)
port2.CalcPort(str(SIM_PATH), freq, ref_impedance=50.0)

uf_inc_1 = np.asarray(port1.uf_inc)
uf_ref_1 = np.asarray(port1.uf_ref)
uf_ref_2 = np.asarray(port2.uf_ref)

print()
print("=" * 72)
print("PORT VALIDATION")
print("=" * 72)
print(f"max |uf_inc_1|     : {np.max(np.abs(uf_inc_1)):.6e}")
print(f"max |uf_ref_1|     : {np.max(np.abs(uf_ref_1)):.6e}")
print(f"max |uf_ref_2|     : {np.max(np.abs(uf_ref_2)):.6e}")

if not np.all(np.isfinite(uf_inc_1)):
    raise RuntimeError("FAIL: incident wave contains NaN/Inf.")
if not np.all(np.isfinite(uf_ref_1)):
    raise RuntimeError("FAIL: reflected wave contains NaN/Inf.")
if not np.all(np.isfinite(uf_ref_2)):
    raise RuntimeError("FAIL: Port 2 response contains NaN/Inf.")

if np.max(np.abs(uf_inc_1)) == 0.0:
    raise RuntimeError("FAIL: Port 1 excitation is exactly zero.")

if np.max(np.abs(uf_ref_2)) == 0.0:
    raise RuntimeError("FAIL: Port 2 response is exactly zero.")

s11 = uf_ref_1 / uf_inc_1
s21 = uf_ref_2 / uf_inc_1
s11_db = 20*np.log10(np.maximum(np.abs(s11), 1e-15))
s21_db = 20*np.log10(np.maximum(np.abs(s21), 1e-15))

print(f"S11 max [dB]       : {np.max(s11_db):.3f}")
print(f"S11 min [dB]       : {np.min(s11_db):.3f}")
print(f"S21 max [dB]       : {np.max(s21_db):.3f}")
print(f"S21 min [dB]       : {np.min(s21_db):.3f}")
print("RESULT             : PASS")
print("=" * 72)

csv_path = RESULTS_PATH / "port_validation.csv"
np.savetxt(
    csv_path,
    np.column_stack([freq, s11_db, s21_db]),
    delimiter=",",
    header="frequency_Hz,S11_dB,S21_dB",
    comments="",
)

plot_path = RESULTS_PATH / "port_validation.png"
plt.figure(figsize=(9, 5))
plt.plot(freq/1e9, s11_db, label="S11")
plt.plot(freq/1e9, s21_db, label="S21")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.title("openEMS Microstrip Port Validation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=200)
plt.close()

print(f"CSV saved           : {csv_path}")
print(f"Plot saved          : {plot_path}")
