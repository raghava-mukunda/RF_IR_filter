#!/usr/bin/env python
"""
Iteration 05 low-pass filter: CSXCAD/openEMS FDTD baseline + optional geometry search.

This script rebuilds the Iteration 05 hybrid model from iter05_openems_design.xml.
It does NOT convert the proprietary AEDT project automatically.

Modes:
  --mode build-only  Build/export XML but do not run FDTD.
  --mode baseline    Run one 0.1 GHz..FMAX FDTD solution (default FMAX=30 GHz).
  --mode optimize    Run a limited width sweep; each candidate is a real FDTD run.

Examples (PowerShell, from the folder containing this script/config):
  .\\.venv\\Scripts\\python.exe .\\iter05_openems_fdtd_optimize.py --mode build-only
  .\\.venv\\Scripts\\python.exe .\\iter05_openems_fdtd_optimize.py --mode baseline
  .\\.venv\\Scripts\\python.exe .\\iter05_openems_fdtd_optimize.py --mode optimize --max-evals 5
  .\\.venv\\Scripts\\python.exe .\\iter05_openems_fdtd_optimize.py --mode baseline --fmax-ghz 100 --n-freq 2001

OpenEMS/CSXCAD must be installed. This script is designed to bootstrap the user's
Windows portable package via OPENEMS_ROOT or --openems-root. The local environment
in which this script was authored did not have openEMS/CSXCAD installed, so the
actual FDTD solve must be run on the user's machine.

Modelling notes:
  * Filter body remains 65.3 x 5.0 x 4.6 mm. 7 mm uniform feed extensions are
    added outside each measurement plane so PML does not coincide with the ports.
  * The 8 series inductors and 7 total shunt capacitances use the supplied Iter05
    values. Each shunt capacitance is split as C/2 to top and bottom PEC.
  * Uses stripline mode ports (x propagation, z electric field), not a multi-cell
    lumped port box. FDTD is blocked on openEMS < 0.37 because its solver does not
    reliably implement the series-inductor model required by this circuit.
  * First pass uses PEC for the center strip and enclosure, plus ideal lumped L/C.
    This isolates topology/coupling first; it is not a finite-loss copper model.
  * Optimization is a deterministic coarse sweep of center-strip width only. It
    does not claim to be a manufacturing optimizer and does not tune L/C values.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import re
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "iter05_openems_design.xml"
DEFAULT_OUT = HERE / "iter05_openems_runs"
_DLL_DIR_HANDLES = []


def _version_tuple_in_text(text: str):
    match = re.search(r'(?<!\d)v?(\d+)\.(\d+)\.(\d+)', text, re.I)
    return tuple(map(int, match.groups())) if match else None


def bootstrap_openems(openems_root: str | None):
    """Import a coherent openEMS installation without mixing its DLLs and wheels.

    Priority:
      1. If --openems-root or OPENEMS_ROOT is explicitly provided, load only that
         bundle and enforce that its Python extensions come from the same bundle.
      2. Otherwise, use the already-activated Python environment as-is. Do not
         silently attach DLL directories from an arbitrary Desktop bundle to venv
         extensions (the previous behavior caused the reported version-mixing error).
      3. Only if normal imports fail, search a portable Desktop bundle as fallback.

    The solver-version compatibility gate is separate and blocks baseline/optimize
    if series-inductor support cannot be verified. build-only is allowed to inspect
    and export geometry without asserting that a full FDTD solve is safe.
    """
    import inspect

    def import_active_environment():
        from CSXCAD import ContinuousStructure  # type: ignore
        from openEMS import openEMS  # type: ignore
        return ContinuousStructure, openEMS

    def module_locations(ContinuousStructure, openEMSClass):
        locations = []
        for cls in (ContinuousStructure, openEMSClass):
            try:
                locations.append(Path(inspect.getfile(cls)).resolve())
            except (TypeError, OSError):
                pass
        return locations

    explicit = openems_root or os.environ.get("OPENEMS_ROOT")
    if not explicit:
        # In a venv, openEMS/CSXCAD may have been installed from matching wheels.
        # Use those first and do not inject a separate portable package's DLLs.
        try:
            ContinuousStructure, openEMSClass = import_active_environment()
            locations = module_locations(ContinuousStructure, openEMSClass)
            print("Using openEMS/CSXCAD from the active Python environment (no portable bundle DLL paths injected).")
            for path in locations:
                print(f"  Python binding: {path}")
            return ContinuousStructure, openEMSClass
        except Exception as active_error:
            # Fall through to portable-bundle discovery only when active imports fail.
            for name in list(sys.modules):
                if name == "CSXCAD" or name.startswith("CSXCAD.") or name == "openEMS" or name.startswith("openEMS."):
                    sys.modules.pop(name, None)
            active_import_error = active_error
    else:
        active_import_error = None

    desktop = Path.home() / "Desktop"
    candidates: list[Path] = []
    if explicit:
        requested = Path(explicit).expanduser()
        if requested.name.lower() == "openems":
            candidates = [requested]
        elif (requested / "openEMS").is_dir():
            candidates = [requested / "openEMS", requested]
        else:
            candidates = [requested]
    else:
        bundles = list(desktop.glob("openEMS_x64*/")) if desktop.exists() else []
        bundles.sort(
            key=lambda item: _version_tuple_in_text(item.name) or (0, 0, 0),
            reverse=True,
        )
        for bundle in bundles:
            candidates.extend([bundle / "openEMS", bundle])

    seen: set[str] = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate.resolve())
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)
    selected = None
    for candidate in unique_candidates:
        if not candidate.is_dir():
            continue
        if (candidate / "Python").is_dir() or (candidate / "bin").is_dir() or any(candidate.glob("*.pyd")):
            selected = candidate.resolve()
            break
    if selected is None:
        searched = "\n".join(f"  - {p}" for p in unique_candidates) or "  (no portable bundles found)"
        detail = f" Active-environment import error: {active_import_error}" if active_import_error else ""
        raise RuntimeError(
            "Could not import openEMS/CSXCAD from the active Python environment and could not locate "
            "a portable bundle. Pass --openems-root to a matching bundle or set OPENEMS_ROOT.\n"
            f"Paths checked:\n{searched}{detail}"
        )

    # Only explicit/fallback portable bundle selection reaches this point; add its
    # DLL paths and Python paths together, then verify both extensions came from it.
    for path in (selected, selected / "bin", selected / "Python"):
        if path.exists() and hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(path)))
            except (OSError, FileNotFoundError):
                pass

    import_paths = [selected, selected / "Python", selected / "Python" / "CSXCAD"]
    for path in reversed(import_paths):
        if path.exists():
            path_string = str(path)
            if path_string in sys.path:
                sys.path.remove(path_string)
            sys.path.insert(0, path_string)

    try:
        ContinuousStructure, openEMSClass = import_active_environment()
    except Exception as exc:
        raise RuntimeError(
            "Could not import openEMS/CSXCAD from the selected bundle "
            f"{selected}. Ensure the bundle's Python extensions match this Python version. "
            f"Original import error: {exc}"
        ) from exc

    locations = module_locations(ContinuousStructure, openEMSClass)
    root_text = str(selected).casefold().rstrip("\\/") + os.sep.casefold()
    outside = [str(path) for path in locations if not str(path).casefold().startswith(root_text)]
    # Some bundle Python layouts place extensions in a sibling Python/ directory.
    selected_text = str(selected).casefold().rstrip("\\/")
    outside = [path for path in outside if not path.casefold().startswith(selected_text + os.sep.casefold())]
    if outside:
        raise RuntimeError(
            "Version-mixing guard: the selected portable bundle was requested, but imported Python "
            f"extensions are outside it. Selected bundle: {selected}. Loaded modules: {outside}. "
            "No FDTD run should be attempted. Install the wheels from this bundle into the active venv "
            "or use the venv installation without --openems-root."
        )

    print(f"Loaded coherent portable openEMS/CSXCAD bundle: {selected}")
    for path in locations:
        print(f"  Python binding: {path}")
    return ContinuousStructure, openEMSClass


def preflight_inductor_support(ContinuousStructure):
    """Fail before meshing if the installed bindings cannot model a series L."""
    csx = ContinuousStructure()
    try:
        # CSXCAD's Python API is keyword-only after the property name.
        # Passing "x" positionally raises: AddLumpedElement() takes exactly 1
        # positional argument (2 given), before L/LEtype support is even tested.
        test_l = csx.AddLumpedElement(
            "__lpf_inductor_api_preflight__",
            ny="x",
            L=1.0e-9,
            caps=False,
            LEtype=1,
        )
        if test_l is None:
            raise RuntimeError("AddLumpedElement returned None for the series-inductor test.")
        if hasattr(test_l, "GetInductance"):
            actual_l = float(test_l.GetInductance())
            if not math.isclose(actual_l, 1.0e-9, rel_tol=1e-6, abs_tol=1e-18):
                raise RuntimeError(f"Series inductance read-back mismatch: {actual_l} H")
        if hasattr(test_l, "GetLEtype"):
            actual_type = int(test_l.GetLEtype())
            if actual_type != 1:
                raise RuntimeError(f"Series RLC topology read-back mismatch: LEtype={actual_type}")
    except Exception as exc:
        raise RuntimeError(
            "PRE-FLIGHT FAILED: the installed CSXCAD Python binding did not "
            "accept or preserve a series lumped inductor (L=1 nH, LEtype=1) "
            "when called with the documented keyword-only API (ny='x'). No "
            "FDTD run was started. The previous TypeError was caused by this "
            "script passing the direction as an extra positional argument; this "
            "corrected test now separates that call-signature issue from actual "
            "inductor/LEtype support. openEMS documentation says full series-RLC "
            "support was added in 0.37; if this test still fails, use a compatible "
            "build or implement physical distributed inductors. Underlying error: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _openems_version_evidence(ContinuousStructure, openEMSClass):
    """Return a detected openEMS version tuple and the evidence used.

    Version numbers are taken only from a versioned openEMS bundle name or an
    explicit module version constant, not from the host Python path.
    """
    import inspect

    module_paths = []
    module_versions = []
    for cls in (openEMSClass, ContinuousStructure):
        try:
            module_paths.append(str(inspect.getfile(cls)))
        except (TypeError, OSError):
            pass
        top = cls.__module__.split('.')[0]
        mod = sys.modules.get(top)
        if mod is not None:
            mod_path = getattr(mod, '__file__', None)
            if mod_path:
                module_paths.append(str(mod_path))
            for attr in ('__version__', 'VERSION', 'version'):
                value = getattr(mod, attr, None)
                if isinstance(value, (str, int, float)):
                    module_versions.append((str(value), f'{top}.{attr}'))

    # The portable package name has the form openEMS_x64_v0.37.0-rc3_msvc.
    # Parse that segment explicitly; do not accidentally read Python 3.13.2 as
    # the solver version from a virtual-environment path.
    for value in module_paths + [str(x) for x in sys.path if isinstance(x, str)]:
        match = re.search(r'openems_x64[^\\/]*?v?(\d+)\.(\d+)\.(\d+)', value, re.I)
        if match:
            return tuple(map(int, match.groups())), value

    for value, source in module_versions:
        match = re.match(r'^\s*v?(\d+)\.(\d+)\.(\d+)', value, re.I)
        if match:
            return tuple(map(int, match.groups())), f'{source}={value}'

    # Installed wheels may expose no __version__ attribute; inspect distribution
    # metadata as a secondary source rather than misreading Python's version/path.
    try:
        from importlib import metadata as importlib_metadata
        for dist_name in ("openEMS", "CSXCAD"):
            try:
                dist_version = importlib_metadata.version(dist_name)
            except importlib_metadata.PackageNotFoundError:
                continue
            match = re.match(r'^\s*v?(\d+)\.(\d+)\.(\d+)', dist_version, re.I)
            if match:
                return tuple(map(int, match.groups())), f'distribution {dist_name}={dist_version}'
    except Exception:
        pass

    evidence = '; '.join(module_paths[:6]) or 'no module paths exposed'
    if module_versions:
        evidence += '; module version constants: ' + ', '.join(f'{k}={v}' for v, k in module_versions)
    return None, evidence


def require_series_inductor_solver_support(ContinuousStructure, openEMSClass,
                                           declared_version: str | None = None):
    """Stop before FDTD unless the loaded solver release includes series-L support.

    CSXCAD can accept and read back L/LEtype on a build whose FDTD executable does
    not implement the inductor correctly. Therefore the object-level preflight is
    not, by itself, sufficient to authorize a simulation.
    """
    detected, evidence = _openems_version_evidence(ContinuousStructure, openEMSClass)
    if detected is not None:
        version = detected
        source = evidence
    elif declared_version:
        nums = re.match(r'^\s*v?(\d+)\.(\d+)\.(\d+)', declared_version, re.I)
        if not nums:
            raise RuntimeError(
                f"Invalid --openems-version value {declared_version!r}; expected e.g. 0.37.0-rc3."
            )
        version = tuple(map(int, nums.groups()))
        source = f"user-supplied --openems-version={declared_version} (could not infer from loaded module path)"
    else:
        raise RuntimeError(
            "FDTD RUN BLOCKED: the loaded openEMS version could not be verified, so this script "
            "will not risk another unstable multi-million-cell solve. Use a complete, matching "
            "openEMS/CSXCAD v0.37.0-rc3-or-newer bundle (series-RLC support is required), or "
            "pass --openems-version 0.37.0-rc3 only after verifying the loaded binary is that version. "
            f"Version evidence: {evidence}"
        )

    if version < (0, 37, 0):
        raise RuntimeError(
            "FDTD RUN BLOCKED BEFORE SOLVE: detected openEMS version "
            f"{version[0]}.{version[1]}.{version[2]} ({source}). This 0.0.36 series-RLC path "
            "does not reliably implement the inductor model; the previous run reached -nan(ind) "
            "at timestep 180. Upgrade the whole openEMS + CSXCAD + Python-binding bundle together "
            "to v0.37.0-rc3 or newer, ensure the venv imports that bundle, and rerun. Do not bypass "
            "this guard for the present build. --mode build-only remains available."
        )
    print(f"Series-inductor solver compatibility gate: openEMS {version[0]}.{version[1]}.{version[2]} ({source})")


def read_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Design config not found: {path}")
    root = ET.parse(path).getroot()
    geom = root.find("Geometry")
    sim = root.find("Simulation")
    ports = root.find("Ports")
    optimization = root.find("Optimization")
    if geom is None or sim is None or ports is None:
        raise ValueError("Config XML must contain Geometry, Simulation, and Ports.")

    def gf(node, attr, default=None):
        value = node.attrib.get(attr, default)
        if value is None:
            raise ValueError(f"Missing XML attribute {attr!r} in {node.tag}")
        return float(value)

    design = {
        "source": root.attrib.get("source", "Iteration05"),
        "length_mm": gf(geom, "length"),
        "outer_w_mm": gf(geom, "outerWidth"),
        "outer_h_mm": gf(geom, "outerHeight"),
        "wall_mm": gf(geom, "wallThickness"),
        "center_w_mm": gf(geom, "centerStripWidth"),
        "center_t_mm": gf(geom, "centerStripThickness"),
        "cap_w_mm": gf(geom, "capacitorPlateWidth"),
        "extension_mm": gf(geom, "feedExtensionEachSide"),
        "pad_y_mm": gf(geom, "airPaddingY"),
        "pad_z_mm": gf(geom, "airPaddingZ"),
        "center_model": geom.attrib.get("centerStripModel", "PEC_initial_pass"),
        "fmin_ghz": gf(sim, "frequencyMinGHz"),
        "fmax_ghz": gf(sim, "frequencyMaxGHz"),
        "n_freq": int(gf(sim, "frequencyPoints")),
        "z0_ohm": gf(sim, "referenceImpedanceOhm"),
        "mesh_mm": gf(sim, "meshStepMM"),
        "nr_ts": int(gf(sim, "maximumTimesteps")),
        "end_criteria": gf(sim, "endCriteria"),
        "pml_cells": int(gf(sim, "pmlCells")),
        "port1_x_mm": gf(ports, "inputXmm"),
        "port2_x_mm": gf(ports, "outputXmm"),
        "width_candidates": [float(v) for v in optimization.attrib["candidateWidthsMM"].split(",")]
            if optimization is not None else [],
        "max_evals": int(optimization.attrib.get("defaultMaximumEvaluations", "5"))
            if optimization is not None else 5,
    }
    design["inner_w_mm"] = design["outer_w_mm"] - 2.0 * design["wall_mm"]
    design["inner_h_mm"] = design["outer_h_mm"] - 2.0 * design["wall_mm"]

    series_parent = root.find("SeriesInductors")
    caps_parent = root.find("ShuntCapacitors")
    if series_parent is None or caps_parent is None:
        raise ValueError("Config XML must include SeriesInductors and ShuntCapacitors.")

    design["inductors"] = [
        {
            "name": e.attrib["name"],
            "element": int(e.attrib["element"]),
            "gap_start_mm": float(e.attrib["gapStartMM"]),
            "gap_end_mm": float(e.attrib["gapEndMM"]),
            "L_H": float(e.attrib["valueH"]),
        }
        for e in series_parent.findall("Inductor")
    ]
    design["capacitors"] = [
        {
            "name": e.attrib["name"],
            "element": int(e.attrib["element"]),
            "x_mm": float(e.attrib["xMM"]),
            "C_F": float(e.attrib["valueF"]),
        }
        for e in caps_parent.findall("Capacitor")
    ]
    if len(design["inductors"]) != 8 or len(design["capacitors"]) != 7:
        raise ValueError("Expected exactly 8 series inductors and 7 shunt capacitors.")
    return design


def axis_lines(low: float, high: float, step: float, special_values) -> np.ndarray:
    """
    Construct a nonuniform mesh that preserves all component/port edges without
    creating tiny sliver cells. Each interval between mandatory coordinates is
    divided into ceil(interval/step) equal cells (cell size <= step).
    """
    mandatory = np.asarray(
        [low, high, *list(special_values)], dtype=float
    )
    mandatory = mandatory[
        (mandatory >= low - 1e-8) & (mandatory <= high + 1e-8)
    ]
    mandatory = np.unique(np.round(mandatory, 9))
    if mandatory.size < 2:
        raise ValueError("Mesh axis needs at least two distinct coordinates.")
    if abs(mandatory[0] - low) > 1e-6 or abs(mandatory[-1] - high) > 1e-6:
        raise ValueError("Mandatory mesh coordinates must include both domain limits.")

    line_parts = []
    for i in range(mandatory.size - 1):
        a = float(mandatory[i])
        b = float(mandatory[i+1])
        interval = b - a
        if interval <= 1e-8:
            continue
        n_cells = max(1, int(math.ceil(interval / step - 1e-12)))
        segment = np.linspace(a, b, n_cells + 1)
        line_parts.append(segment if i == 0 else segment[1:])
    result = np.concatenate(line_parts)
    result[0] = low
    result[-1] = high
    return np.round(result, 9)


def add_box(prop, start, stop):
    # CSXCAD Python API takes the absolute start and stop coordinates in drawing units.
    prop.AddBox(list(map(float, start)), list(map(float, stop)))


def build_model(design: dict, case_dir: Path, ContinuousStructure, openEMSClass,
                center_width_override: float | None = None, fmax_override: float | None = None,
                nfreq_override: int | None = None, build_only: bool = False) -> dict | None:
    d = dict(design)
    if center_width_override is not None:
        d["center_w_mm"] = float(center_width_override)
    if fmax_override is not None:
        d["fmax_ghz"] = float(fmax_override)
        # Keep roughly 0.05 GHz spacing when user overrides max frequency.
        if nfreq_override is None:
            d["n_freq"] = max(601, int(round((d["fmax_ghz"] - d["fmin_ghz"]) / 0.05)) + 1)
    if nfreq_override is not None:
        d["n_freq"] = int(nfreq_override)

    L = d["length_mm"]
    ow = d["outer_w_mm"]
    oh = d["outer_h_mm"]
    wall = d["wall_mm"]
    iw = d["inner_w_mm"]
    ih = d["inner_h_mm"]
    cw = d["center_w_mm"]
    ct = d["center_t_mm"]
    ext = d["extension_mm"]
    pad_y = d["pad_y_mm"]
    pad_z = d["pad_z_mm"]
    capw = d["cap_w_mm"]
    if cw <= 0 or cw >= iw:
        raise ValueError(f"Center conductor width {cw} mm must lie in (0, {iw} mm).")
    if capw <= 0 or capw >= min(0.5, cw):
        raise ValueError("Capacitor plate width is invalid.")
    if d["fmax_ghz"] <= d["fmin_ghz"]:
        raise ValueError("Fmax must exceed Fmin.")
    if not math.isclose(d["port1_x_mm"], 0.0, abs_tol=1e-6):
        raise ValueError("For the current feed layout, inputXmm must be 0.0 mm.")
    if not math.isclose(d["port2_x_mm"], L, abs_tol=1e-6):
        raise ValueError("For the current feed layout, outputXmm must equal the filter length.")

    case_dir.mkdir(parents=True, exist_ok=True)
    sim_dir = case_dir / "simulation"
    # Keep input XMLs/artifacts but remove stale run products only.
    if sim_dir.exists():
        shutil.rmtree(sim_dir)
    sim_dir.mkdir(parents=True, exist_ok=True)

    CSX = ContinuousStructure()
    enclosure = CSX.AddMetal("PEC_Enclosure")
    if d["center_model"].lower().startswith("copper"):
        strip = CSX.AddMaterial("CenterConductor_Copper", epsilon=1.0, mue=1.0, kappa=5.8e7)
        strip_model_used = "finite conductivity copper (skin depth not resolved by current mesh)"
    else:
        strip = CSX.AddMetal("PEC_CenterConductor")
        strip_model_used = "PEC center conductor (initial low-loss topology test)"

    # Simulation padding/line extensions are outside the 65.3 mm physical filter.
    x_min = -ext
    x_max = L + ext
    y_min = -ow / 2.0 - pad_y
    y_max = ow / 2.0 + pad_y
    z_min = -oh / 2.0 - pad_z
    z_max = oh / 2.0 + pad_z
    yi = iw / 2.0
    zi = ih / 2.0

    # Enclosure walls: each wall is a PEC solid; x ends are left open for ports.
    # Top/bottom walls run across full outer width; side walls touch but do not overlap them.
    add_box(enclosure, [x_min, -ow/2, zi], [x_max, ow/2, oh/2])
    add_box(enclosure, [x_min, -ow/2, -oh/2], [x_max, ow/2, -zi])
    add_box(enclosure, [x_min, -ow/2, -zi], [x_max, -yi, zi])
    add_box(enclosure, [x_min, yi, -zi], [x_max, ow/2, zi])

    # Uniform center-conductor feed extensions outside the measurement planes.
    add_box(strip, [x_min, -cw/2, -ct/2], [0.0, cw/2, ct/2])
    add_box(strip, [L, -cw/2, -ct/2], [x_max, cw/2, ct/2])

    # Nine non-overlapping conductor nodes, separated by the eight prescribed L gaps.
    previous_x = 0.0
    node_ranges = []
    for elem in sorted(d["inductors"], key=lambda item: item["gap_start_mm"]):
        gx0 = elem["gap_start_mm"]
        gx1 = elem["gap_end_mm"]
        if gx0 <= previous_x or gx1 <= gx0 or gx1 >= L:
            raise ValueError(f"Invalid series gap geometry for {elem['name']}: {gx0}..{gx1}")
        node_ranges.append((previous_x, gx0))
        previous_x = gx1
    node_ranges.append((previous_x, L))
    for i, (x0, x1) in enumerate(node_ranges, start=1):
        if x1 <= x0:
            raise ValueError(f"Empty center-conductor node {i}: {x0}..{x1}")
        add_box(strip, [x0, -cw/2, -ct/2], [x1, cw/2, ct/2])

    # Series inductors across the gaps. L is in henries, with a series RLC model.
    for elem in d["inductors"]:
        try:
            lump = CSX.AddLumpedElement(
                elem["name"], ny="x", L=elem["L_H"], caps=True, LEtype=1
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to create {elem['name']} using inductive lumped elements. "
                "No FDTD run was started. Verify openEMS >=0.37/API support. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        add_box(
            lump,
            [elem["gap_start_mm"], -cw/2, -ct/2],
            [elem["gap_end_mm"], cw/2, ct/2],
        )

    # Each total shunt C is split into C/2 to the top and bottom enclosure walls.
    for elem in d["capacitors"]:
        xc = elem["x_mm"]
        cbranch = elem["C_F"] / 2.0
        top = CSX.AddLumpedElement(f'{elem["name"]}_Top', ny="z", C=cbranch, caps=True)
        add_box(
            top,
            [xc-capw/2, -cw/2, ct/2],
            [xc+capw/2, cw/2, zi],
        )
        bottom = CSX.AddLumpedElement(f'{elem["name"]}_Bottom', ny="z", C=cbranch, caps=True)
        add_box(
            bottom,
            [xc-capw/2, -cw/2, -zi],
            [xc+capw/2, cw/2, -ct/2],
        )

    # FDTD axes are in mm; SetDeltaUnit converts them to SI metres.
    # Explicit component/port/wall coordinates are injected to align mesh lines.
    specials_x = [0.0, L, d["port1_x_mm"], d["port2_x_mm"]]
    for e in d["inductors"]:
        specials_x.extend([e["gap_start_mm"], e["gap_end_mm"]])
    for e in d["capacitors"]:
        specials_x.extend([e["x_mm"]-capw/2, e["x_mm"]+capw/2])
    x_lines = axis_lines(x_min, x_max, d["mesh_mm"], specials_x)

    specials_y = [-ow/2, -yi, -cw/2, cw/2, yi, ow/2, -iw/2, iw/2]
    y_lines = axis_lines(y_min, y_max, d["mesh_mm"], specials_y)

    specials_z = [-oh/2, -zi, -ct/2, 0.0, ct/2, zi, oh/2]
    z_lines = axis_lines(z_min, z_max, d["mesh_mm"], specials_z)

    grid = CSX.GetGrid()
    grid.SetLines("x", x_lines)
    grid.SetLines("y", y_lines)
    grid.SetLines("z", z_lines)
    grid.SetDeltaUnit(1e-3)

    # Configure FDTD and stripline ports. The line extensions run from x_min to
    # the input plane and from the output plane to x_max, away from PML.
    fmax = d["fmax_ghz"] * 1e9
    FDTD = openEMSClass(NrTS=d["nr_ts"], EndCriteria=d["end_criteria"])
    FDTD.SetCSX(CSX)
    FDTD.SetBoundaryCond(["PML_8"] * 6)
    # A low-pass S-parameter sweep needs a low-pass Gaussian spectrum spanning
    # DC through fmax. (f0=fmax/2, fc=fmax/2 under-excites the lower band.)
    FDTD.SetGaussExcite(0.0, fmax)

    # The line is a shielded stripline: x is propagation, z is the E-field
    # direction, and the PEC lid/base are the two reference planes. The previous
    # lumped-port box spanned the whole y-z cross-section and behaved as a
    # multi-cell lumped load rather than a transmission-line modal port.
    add_stripline_port = getattr(FDTD, "AddStripLinePort", None)
    if not callable(add_stripline_port):
        raise RuntimeError(
            "This openEMS binding has no AddStripLinePort API; a correct stripline "
            "port cannot be defined. No FDTD run was started. Use a compatible "
            "openEMS Python binding rather than falling back to the old multi-cell lumped port."
        )
    feed_shift = max(0.5, ext - max(1.0, 4.0*d["mesh_mm"]))
    port1 = add_stripline_port(
        1, strip,
        [x_min, -cw/2, 0.0],
        [d["port1_x_mm"], cw/2, 0.0],
        "x", "z", zi,
        excite=1.0, FeedShift=feed_shift,
        MeasPlaneShift=ext, priority=30,
    )
    # Reverse the output port's propagation direction so the wave transmitted
    # toward +x is represented by port2.uf_ref, matching the S21 calculation below.
    port2 = add_stripline_port(
        2, strip,
        [x_max, -cw/2, 0.0],
        [d["port2_x_mm"], cw/2, 0.0],
        "x", "z", zi,
        excite=0.0, MeasPlaneShift=ext, priority=30,
    )

    # Always export geometry for visual inspection. A combined solver+geometry
    # XML is available only in openEMS Python bindings that expose Write2XML on
    # the openEMS instance (notably newer builds). Older bindings can still run
    # through FDTD.Run(), so absence of the optional writer must not abort build-only.
    geom_xml = case_dir / "Iter05_openEMS_geometry.xml"
    solver_xml = case_dir / "Iter05_openEMS_FDTD.xml"
    CSX.Write2XML(str(geom_xml))
    solver_xml_written = False
    write_solver_xml = getattr(FDTD, "Write2XML", None)
    if callable(write_solver_xml):
        try:
            write_solver_xml(str(solver_xml))
            solver_xml_written = solver_xml.exists() and solver_xml.stat().st_size > 0
            if not solver_xml_written:
                print("WARNING: FDTD.Write2XML was present but did not create a non-empty file.")
        except Exception as exc:
            print(
                "WARNING: this openEMS binding could not export a combined solver XML; "
                f"Python FDTD execution remains available. Reason: {type(exc).__name__}: {exc}"
            )
    else:
        print(
            "NOTE: this installed openEMS binding has no FDTD.Write2XML method. "
            "Only geometry XML will be exported; build-only can still succeed, and "
            "FDTD runs use the configured Python objects directly."
        )

    counts = {
        "x_nodes": int(len(x_lines)),
        "y_nodes": int(len(y_lines)),
        "z_nodes": int(len(z_lines)),
        "cells_estimated": int((len(x_lines)-1) * (len(y_lines)-1) * (len(z_lines)-1)),
    }
    print(f"\nCase: {case_dir}")
    print(f"Physical filter: {L:.3f} x {ow:.3f} x {oh:.3f} mm")
    print(f"Simulation domain: x {x_min:.3f}..{x_max:.3f}, y {y_min:.3f}..{y_max:.3f}, z {z_min:.3f}..{z_max:.3f} mm")
    print(f"Center strip: {cw:.3f} x {ct:.3f} mm; material={strip_model_used}")
    print(f"Grid nodes x/y/z = {counts['x_nodes']}/{counts['y_nodes']}/{counts['z_nodes']}; "
          f"interior cells estimate={counts['cells_estimated']:,}")
    print(f"Geometry XML (view in AppCSXCAD): {geom_xml}")
    if solver_xml_written:
        print(f"Self-contained solver XML: {solver_xml}")
    else:
        print("Self-contained solver XML: not available from this openEMS Python binding")

    if build_only:
        return None

    freqs = np.linspace(d["fmin_ghz"] * 1e9, fmax, d["n_freq"])
    print(f"\nStarting one FDTD solve from {d['fmin_ghz']:.3f} to {d['fmax_ghz']:.3f} GHz "
          f"({d['n_freq']} samples); max timesteps={d['nr_ts']:,}.")
    try:
        FDTD.Run(str(sim_dir), cleanup=True, verbose=2)
    except Exception as exc:
        message = str(exc).lower()
        if "nan" in message or "float64" in message or "could not convert string" in message:
            raise RuntimeError(
                "FDTD FAILED WITH NON-FINITE ENERGY / NaN. No S-parameters were calculated or "
                "saved. This indicates solver/model instability, not a CSV parsing problem. "
                "Check the series-inductor solver version, stripline ports, and lumped-element geometry. "
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc
        raise

    port1.CalcPort(str(sim_dir), freqs, ref_impedance=d["z0_ohm"])
    port2.CalcPort(str(sim_dir), freqs, ref_impedance=d["z0_ohm"])
    tiny = 1e-15
    # Normalize the reflected and transmitted waves to the incident wave at Port 1.
    s11 = np.asarray(port1.uf_ref) / np.where(np.abs(port1.uf_inc) > tiny, port1.uf_inc, tiny+0j)
    s21 = np.asarray(port2.uf_ref) / np.where(np.abs(port1.uf_inc) > tiny, port1.uf_inc, tiny+0j)
    s11_db = 20.0*np.log10(np.maximum(np.abs(s11), 1e-15))
    s21_db = 20.0*np.log10(np.maximum(np.abs(s21), 1e-15))

    ideal_s11, ideal_s21 = ideal_ladder_sparams(freqs, d["z0_ohm"], d["inductors"], d["capacitors"])
    ideal_s11_db = 20.0*np.log10(np.maximum(np.abs(ideal_s11), 1e-15))
    ideal_s21_db = 20.0*np.log10(np.maximum(np.abs(ideal_s21), 1e-15))

    csv_path = case_dir / "S_parameters.csv"
    np.savetxt(
        csv_path,
        np.column_stack([
            freqs/1e9, s11.real, s11.imag, np.abs(s11), s11_db,
            s21.real, s21.imag, np.abs(s21), s21_db,
            ideal_s11_db, ideal_s21_db,
        ]),
        delimiter=",",
        header=("Frequency_GHz,S11_real,S11_imag,S11_mag,S11_dB,"
                "S21_real,S21_imag,S21_mag,S21_dB,Ideal_S11_dB,Ideal_S21_dB"),
        comments="",
    )

    # The structure is reciprocal and mirror-symmetric (same ports, mirrored L/C);
    # S12=S21 and S22=S11 are inferred for the 2-port Touchstone export.
    touchstone = case_dir / "S_parameters.s2p"
    with touchstone.open("w", encoding="ascii") as f:
        f.write("! Iteration 05 openEMS FDTD; stripline ports; reciprocal/mirror-symmetric assumption: S12=S21, S22=S11\n")
        f.write("# GHz S RI R 50\n")
        for i, freq in enumerate(freqs):
            vals = [s11[i].real, s11[i].imag,
                    s21[i].real, s21[i].imag,
                    s21[i].real, s21[i].imag,
                    s11[i].real, s11[i].imag]
            f.write(f"{freq/1e9:.9g} " + " ".join(f"{v:.12e}" for v in vals) + "\n")

    plot_path = case_dir / "S11_S21_openEMS_vs_ideal.png"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.plot(freqs/1e9, s21_db, label="openEMS S21", linewidth=1.3)
    ax.plot(freqs/1e9, s11_db, label="openEMS S11", linewidth=1.0)
    ax.plot(freqs/1e9, ideal_s21_db, "--", label="ideal LC S21", linewidth=1.0)
    ax.axvline(10.0, color="grey", linestyle=":", linewidth=1)
    ax.axhline(-60.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(0, d["fmax_ghz"])
    ax.set_ylim(-120, 5)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Iteration 05 hybrid LPF — openEMS FDTD vs ideal LC reference")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    f_ghz = freqs/1e9
    passband = (f_ghz >= d["fmin_ghz"]) & (f_ghz <= min(9.0, d["fmax_ghz"]))
    stopband = (f_ghz >= 12.0) & (f_ghz <= min(30.0, d["fmax_ghz"]))
    f3 = first_minus3db_crossing(f_ghz, s21_db)
    pass_min = float(np.min(s21_db[passband])) if np.any(passband) else None
    pass_max = float(np.max(s21_db[passband])) if np.any(passband) else None
    worst_stop = float(np.max(s21_db[stopband])) if np.any(stopband) else None
    worst_stop_f = float(f_ghz[stopband][np.argmax(s21_db[stopband])]) if np.any(stopband) else None
    pb_rms = float(np.sqrt(np.mean((s21_db[passband]-ideal_s21_db[passband])**2))) if np.any(passband) else None
    stop_violation = max(0.0, worst_stop + 60.0) if worst_stop is not None else None
    score = float(2.0*pb_rms + 10.0*(abs(f3-10.0) if f3 is not None else 10.0) +
                  2.0*(stop_violation if stop_violation is not None else 100.0))

    summary = {
        "case": case_dir.name,
        "model_status": "hybrid full-wave FDTD with ideal lumped L/C; not manufacturing ready",
        "physical_length_mm": L,
        "outer_envelope_mm": [L, ow, oh],
        "center_strip_width_mm": cw,
        "center_strip_thickness_mm": ct,
        "center_strip_model": strip_model_used,
        "frequency_range_GHz": [d["fmin_ghz"], d["fmax_ghz"]],
        "frequency_samples": int(d["n_freq"]),
        "estimated_cells_before_PML": counts["cells_estimated"],
        "f3dB_first_crossing_GHz": f3,
        "passband_S21_min_dB_0p1_to_9": pass_min,
        "passband_S21_max_dB_0p1_to_9": pass_max,
        "passband_RMSE_vs_ideal_dB": pb_rms,
        "worst_S21_dB_12_to_30_GHz": worst_stop,
        "worst_stopband_frequency_GHz": worst_stop_f,
        "stopband_violation_above_minus60_dB": stop_violation,
        "optimization_score_lower_is_better": score,
        "geometry_xml": str(geom_xml),
        "solver_xml": str(solver_xml) if solver_xml_written else None,
        "sparameters_csv": str(csv_path),
        "touchstone_s2p": str(touchstone),
        "plot_png": str(plot_path),
        "note": "Only S11/S21 are directly simulated in this excitation; S12=S21 and S22=S11 are inferred by reciprocity and geometric symmetry for the exported S2P.",
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nFDTD summary")
    print(f"  First -3 dB crossing: {f3 if f3 is not None else 'not found'} GHz")
    print(f"  Passband min/max S21: {pass_min} / {pass_max} dB")
    print(f"  Worst S21, 12-30 GHz: {worst_stop} dB at {worst_stop_f} GHz")
    print(f"  Score (lower is better): {score:.4f}")
    print(f"  CSV: {csv_path}")
    print(f"  Touchstone: {touchstone}")
    print(f"  Plot: {plot_path}")
    print(f"  Summary: {case_dir / 'summary.json'}")

    # Drop solver objects between candidate runs to reduce memory retained by Python.
    del FDTD, CSX, port1, port2
    gc.collect()
    return summary


def first_minus3db_crossing(freq_ghz: np.ndarray, response_db: np.ndarray) -> float | None:
    below = np.where(response_db <= -3.0)[0]
    for idx in below:
        if idx == 0:
            return float(freq_ghz[0])
        if response_db[idx-1] > -3.0:
            y0, y1 = response_db[idx-1], response_db[idx]
            x0, x1 = freq_ghz[idx-1], freq_ghz[idx]
            if abs(y1-y0) < 1e-15:
                return float(x1)
            return float(x0 + (-3.0-y0)*(x1-x0)/(y1-y0))
    return None


def ideal_ladder_sparams(freqs: np.ndarray, z0: float, inductors, capacitors):
    """Ideal ABCD calculation of the supplied L1-C2-...-L15 ladder."""
    cap_by_element = {int(c["element"]): float(c["C_F"]) for c in capacitors}
    ind_by_element = {int(l["element"]): float(l["L_H"]) for l in inductors}
    s11 = np.empty(freqs.shape, dtype=complex)
    s21 = np.empty(freqs.shape, dtype=complex)
    for j, freq in enumerate(freqs):
        w = 2.0*np.pi*freq
        M = np.eye(2, dtype=complex)
        for element in range(1, 16):
            if element % 2 == 1:
                z = 1j*w*ind_by_element[element]
                elem_M = np.array([[1.0, z], [0.0, 1.0]], dtype=complex)
            else:
                y = 1j*w*cap_by_element[element]
                elem_M = np.array([[1.0, 0.0], [y, 1.0]], dtype=complex)
            M = M @ elem_M
        A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
        den = A + B/z0 + C*z0 + D
        s11[j] = (A + B/z0 - C*z0 - D) / den
        s21[j] = 2.0 / den
    return s11, s21


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Input parameter XML")
    parser.add_argument("--openems-root", default=None,
                        help="Path to packaged openEMS folder (or set OPENEMS_ROOT)")
    parser.add_argument("--openems-version", default=None,
                        help="Only when module paths do not reveal a version: declare the verified loaded version, e.g. 0.37.0-rc3")
    parser.add_argument("--output-root", default=str(DEFAULT_OUT), help="Folder for XML/results")
    parser.add_argument("--mode", choices=["build-only", "baseline", "optimize"], default="baseline")
    parser.add_argument("--fmax-ghz", type=float, default=None,
                        help="Override frequency maximum; default from config (30 GHz). Use 100 to follow reference.")
    parser.add_argument("--n-freq", type=int, default=None,
                        help="Override S-parameter samples; use 2001 with --fmax-ghz 100.")
    parser.add_argument("--max-evals", type=int, default=None,
                        help="Maximum widths tested in --mode optimize; default from XML (5).")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    design = read_config(config_path)
    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    ContinuousStructure, openEMSClass = bootstrap_openems(args.openems_root)
    preflight_inductor_support(ContinuousStructure)
    if args.mode in ("baseline", "optimize"):
        require_series_inductor_solver_support(
            ContinuousStructure, openEMSClass, declared_version=args.openems_version
        )

    if args.mode in ("build-only", "baseline"):
        case_dir = out_root / "baseline"
        summary = build_model(
            design, case_dir, ContinuousStructure, openEMSClass,
            fmax_override=args.fmax_ghz,
            nfreq_override=args.n_freq,
            build_only=(args.mode == "build-only"),
        )
        if args.mode == "build-only":
            print("\nBuild-only complete: geometry XML exported; no FDTD solve was run.")
        else:
            print("\nBaseline FDTD run complete.")
        return

    max_evals = args.max_evals or design["max_evals"]
    widths = design["width_candidates"]
    # Start with nominal 2.2 mm, then expand symmetrically across the configured grid.
    nominal = design["center_w_mm"]
    widths = sorted(set(widths), key=lambda w: (abs(w - nominal), w))
    widths = widths[:max_evals]
    if not widths:
        raise RuntimeError("No optimization width candidates are configured.")

    records = []
    print("\nStarting limited deterministic openEMS geometry sweep.")
    print("This performs one FDTD solve per width; it is not a zero-cost optimizer.")
    for width in widths:
        name = f"width_{width:.2f}mm".replace(".", "p")
        case_dir = out_root / "optimization" / name
        try:
            result = build_model(
                design, case_dir, ContinuousStructure, openEMSClass,
                center_width_override=width,
                fmax_override=args.fmax_ghz,
                nfreq_override=args.n_freq,
                build_only=False,
            )
            if result:
                records.append(result)
        except Exception as exc:
            error_record = {
                "case": name, "center_strip_width_mm": width,
                "error": f"{type(exc).__name__}: {exc}",
            }
            records.append(error_record)
            (case_dir / "failed_run.json").write_text(json.dumps(error_record, indent=2), encoding="utf-8")
            print(f"\nCandidate {name} failed: {error_record['error']}")
            print("Stopping after first failed candidate to avoid wasting compute.")
            break

    results_path = out_root / "optimization_results.json"
    results_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    valid = [r for r in records if "optimization_score_lower_is_better" in r]
    if valid:
        best = min(valid, key=lambda r: r["optimization_score_lower_is_better"])
        (out_root / "best_geometry_candidate.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
        print("\nOptimization summary")
        print(f"  Best width: {best['center_strip_width_mm']:.3f} mm")
        print(f"  Score: {best['optimization_score_lower_is_better']:.4f}")
        print(f"  Best case directory: {best['case']}")
        print(f"  Results: {results_path}")
        print("Note: this coarse width-only sweep is not a manufacturing-ready optimization.")
    else:
        print(f"\nNo candidate completed. See {results_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:", error, file=sys.stderr)
        if os.environ.get("LPF_OPENEMS_DEBUG", "").lower() in ("1", "true", "yes"):
            traceback.print_exc()
        sys.exit(2)
