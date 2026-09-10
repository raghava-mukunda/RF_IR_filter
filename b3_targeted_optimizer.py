#!/usr/bin/env python
"""Targeted full-wave EM optimizer for the B3 cascade."""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_RUNNER = PROJECT_ROOT / "cheb_b3_butterworth_cascade_v1.py"
TEMPLATE_RUNNER = PROJECT_ROOT / "_b3_targeted_candidate_runner.py"
CAMPAIGN_DIR = PROJECT_ROOT / "results" / "em" / "b3_targeted_optimizer"
HISTORY_PATH = CAMPAIGN_DIR / "history.jsonl"
BEST_PATH = CAMPAIGN_DIR / "best_geometry.json"
LOG_DIR = CAMPAIGN_DIR / "logs"
GRID = 0.25
BASE_CHEB = [1.00, .75, 1.25, 1.00, 1.25, .75, 1.00]
BASE_BUTTER = [.50, 1.25, 2.25, 2.50, 2.25, 1.25, .50]
BASE = {"cheb_scale":1.0,"butter_scale":1.0,"butter_center_delta":0.0,"gap":5.0}
BOUNDS = {"cheb_scale":(.75,1.0),"butter_scale":(.75,1.10),"butter_center_delta":(-.75,.75),"gap":(2.5,7.5)}
STAGES = [
 {"name":"COARSE","steps":{"cheb_scale":.0625,"butter_scale":.0625,"butter_center_delta":.25,"gap":1.25}},
 {"name":"FINE","steps":{"cheb_scale":.03125,"butter_scale":.03125,"butter_center_delta":.25,"gap":.625}},
]

def snap(x): return round(x/GRID)*GRID
def clamp(k,x): return max(BOUNDS[k][0],min(BOUNDS[k][1],x))
def geometry_from(p):
    """Return grid-quantized section lengths and a grid-valid cascade gap.

    Every section is constrained to the 0.25-mm Cartesian grid.  Centering the
    complete cascade additionally requires the *total* x length to be a
    multiple of 0.50 mm; otherwise +/- total_length/2 lands on a half-grid
    coordinate such as 18.875 mm and the candidate runner correctly rejects it.
    We therefore snap the requested gap to the nearest 0.25-mm value that also
    makes the complete cascade centerable, while staying inside the gap bounds.
    """
    c=[snap(v*p["cheb_scale"]) for v in BASE_CHEB]
    b=[snap(v*p["butter_scale"]) for v in BASE_BUTTER]
    b[3]=snap(b[3]+p["butter_center_delta"])
    if min(c+b)<GRID:
        raise ValueError("zero-length section")

    requested_gap = snap(p["gap"])
    # Each stage includes 8 x 1-mm transitions.  The full cascade is
    # centerable when (sum(c)+sum(b)+16mm+gap) is a multiple of 0.50 mm.
    base_total = sum(c) + sum(b) + 16.0
    candidates = []
    for n in range(int(round(BOUNDS["gap"][0]/GRID)),
                   int(round(BOUNDS["gap"][1]/GRID)) + 1):
        g = n * GRID
        total = base_total + g
        if abs(total/0.50 - round(total/0.50)) < 1e-9:
            candidates.append(g)
    if not candidates:
        raise ValueError("No grid-valid cascade gap exists inside bounds")
    gap = min(candidates, key=lambda g: (abs(g-requested_gap), g))
    return c,b,gap

def tag_for(p,rid,actual_gap=None):
    def f(x): return f"{x:.4f}".replace('-','m').replace('.','p')
    gap = p["gap"] if actual_gap is None else actual_gap
    return f"run_{rid:04d}_cs{f(p['cheb_scale'])}_bs{f(p['butter_scale'])}_cd{f(p['butter_center_delta'])}_gap{f(gap)}"

def objective(m):
    return (20*max(0,-1.5-m['passband_min_1_9p5_dB']) +
            15*max(0,-3.0-m['s21_10GHz_dB']) +
            5*max(0,m['max_s21_14_18_dB']+30) +
            3*max(0,m['max_s21_20_100_dB']+40))

def make_template():
    if not BASE_RUNNER.is_file(): raise FileNotFoundError(f"Missing {BASE_RUNNER}")
    t=BASE_RUNNER.read_text(encoding='utf-8')
    t=t.replace('W_HIGH = 0.50\nW_LOW = 12.50','W_HIGH = float(os.environ.get("B3_W_HIGH","0.50"))\nW_LOW = float(os.environ.get("B3_W_LOW","12.50"))',1)
    t=t.replace('ANALYSIS_START = 1.0e9\nANALYSIS_STOP = 100.0e9\nN_FREQ = 1981','ANALYSIS_START = 1.0e9\nANALYSIS_STOP = float(os.environ.get("B3_ANALYSIS_STOP_GHZ","100.0"))*1e9\nN_FREQ = int(os.environ.get("B3_N_FREQ","1981"))',1)
    t=t.replace('PROJECT_ROOT = Path(__file__).resolve().parent\n\nSIM_PATH = PROJECT_ROOT / "simulations" / "openems" / "results" / "cheb_b3_butterworth_cascade_v1"\nRESULTS_PATH = PROJECT_ROOT / "results" / "em" / "cheb_b3_butterworth_cascade_v1"\nPLOTS_PATH = PROJECT_ROOT / "results" / "plots"','PROJECT_ROOT = Path(__file__).resolve().parent\n\n_tag = os.environ.get("B3_TAG","cheb_b3_butterworth_cascade_v1")\nSIM_PATH = PROJECT_ROOT / "simulations" / "openems" / "results" / _tag\nRESULTS_PATH = PROJECT_ROOT / "results" / "em" / _tag\nPLOTS_PATH = PROJECT_ROOT / "results" / "plots" / _tag',1)
    t=t.replace('CHEB_SECTION_LENGTHS = [1.00, 0.75, 1.25, 1.00, 1.25, 0.75, 1.00]','CHEB_SECTION_LENGTHS = [float(v) for v in os.environ.get("B3_CHEB_LENGTHS","1.00,0.75,1.25,1.00,1.25,0.75,1.00").split(",")]',1)
    old='BUTTER_SECTION_LENGTHS = butterworth_section_lengths()\n# Center the complete cascade on the 0.25-mm Cartesian grid. The raw\n# prototype quantization gives 38.25 mm total; add one 0.25 mm grid cell to\n# the central shunt section so the overall 38.50 mm cascade has endpoints\n# at +/-19.25 mm.\nBUTTER_SECTION_LENGTHS[3] += GRID'
    new='if os.environ.get("B3_BUTTER_LENGTHS"):\n    BUTTER_SECTION_LENGTHS = [float(v) for v in os.environ["B3_BUTTER_LENGTHS"].split(",")]\nelse:\n    BUTTER_SECTION_LENGTHS = butterworth_section_lengths()\n    BUTTER_SECTION_LENGTHS[3] += GRID'
    if old not in t: raise RuntimeError('Butterworth length block not found')
    t=t.replace(old,new,1)
    t=t.replace('CASCADE_GAP = 5.00','CASCADE_GAP = float(os.environ.get("B3_CASCADE_GAP","5.00"))',1)
    anchor='attenuation20 = -float(S21_dB[i20])\n'
    metrics='''attenuation20 = -float(S21_dB[i20])\n\n_res14_18 = (freq >= 14e9) & (freq <= 18e9)\n_res20_100 = (freq >= 20e9) & (freq <= min(100e9, ANALYSIS_STOP))\n_pass_1_9p5 = (freq >= 1e9) & (freq <= 9.5e9)\nmetrics = {\n    "tag": os.environ.get("B3_TAG", "candidate"),\n    "cheb_lengths": [float(v) for v in CHEB_SECTION_LENGTHS],\n    "butter_lengths": [float(v) for v in BUTTER_SECTION_LENGTHS],\n    "cascade_gap_mm": float(CASCADE_GAP),\n    "s21_5GHz_dB": float(S21_dB[i5]),\n    "s21_9GHz_dB": float(S21_dB[i9]),\n    "s21_10GHz_dB": float(S21_dB[i10]),\n    "s21_20GHz_dB": float(S21_dB[i20]),\n    "s21_100GHz_dB": float(S21_dB[i100]),\n    "passband_min_1_9p5_dB": float(np.min(S21_dB[_pass_1_9p5])),\n    "max_s21_14_18_dB": float(np.max(S21_dB[_res14_18])),\n    "max_s21_20_100_dB": float(np.max(S21_dB[_res20_100])),\n    "power_min": float(np.min(power)),\n    "power_max": float(np.max(power)),\n}\nmetrics_path = RESULTS_PATH / "metrics.json"\nwith metrics_path.open("w", encoding="utf-8") as _mf:\n    import json as _json\n    _json.dump(metrics, _mf, indent=2)\n    _mf.flush()\n    os.fsync(_mf.fileno())\nprint(f"OPTIMIZER_METRICS={metrics_path}")\nprint("METRICS_EXISTS=" + str(metrics_path.is_file()))\n'''
    if anchor not in t: raise RuntimeError('metrics anchor not found')
    t=t.replace(anchor,metrics,1)
    TEMPLATE_RUNNER.write_text(t,encoding='utf-8')
    print(f'Prepared {TEMPLATE_RUNNER}')

def run_candidate(p,rid,stage):
    cheb,butter,gap=geometry_from(p); tag=tag_for(p,rid,gap)
    log=LOG_DIR/f'{tag}.log'; log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env.update({
      'B3_TAG':str(Path('b3opt')/tag),
      'B3_CHEB_LENGTHS':','.join(f'{x:.2f}' for x in cheb),
      'B3_BUTTER_LENGTHS':','.join(f'{x:.2f}' for x in butter),
      'B3_CASCADE_GAP':f'{gap:.2f}','B3_ANALYSIS_STOP_GHZ':'100.0','B3_N_FREQ':'1981',
      'B3_W_HIGH':'0.50','B3_W_LOW':'12.50'})
    print('\n'+'='*92); print(f'RUN {rid:04d} | {stage}'); print(f'ChebScale={p["cheb_scale"]:.4f} ButterScale={p["butter_scale"]:.4f} CenterDelta={p["butter_center_delta"]:+.2f} mm RequestedGap={p["gap"]:.2f} mm ActualGap={gap:.2f} mm'); print(f'Cheb={cheb}'); print(f'Butter={butter}'); print('Starting full-wave 1–100 GHz EM run...')
    start=time.time()
    with log.open('w',encoding='utf-8',errors='replace') as fh:
      proc=subprocess.Popen([sys.executable,str(TEMPLATE_RUNNER)],cwd=PROJECT_ROOT,env=env,stdout=fh,stderr=subprocess.STDOUT,text=True)
      spin='|/-\\'; k=0; last=''
      while proc.poll() is None:
        try:
          lines=log.read_text(encoding='utf-8',errors='replace').splitlines(); last=lines[-1].strip() if lines else ''
        except Exception: pass
        print(f'\r elapsed {(time.time()-start)/60:6.1f} min {spin[k%4]} {last[:115]:115s}',end='',flush=True); k+=1; time.sleep(1)
    print(); elapsed=time.time()-start
    if proc.returncode!=0:
      raise RuntimeError(f'Candidate failed rc={proc.returncode}. Log={log}\n'+ '\n'.join(log.read_text(encoding='utf-8',errors='replace').splitlines()[-40:]))
    txt=log.read_text(encoding='utf-8',errors='replace'); mm=re.findall(r'^OPTIMIZER_METRICS=(.+)$',txt,re.M)
    if not mm: raise RuntimeError(f'No metrics line in {log}')
    mp=Path(mm[-1].strip())
    if not mp.is_file(): raise RuntimeError(f'Metrics file missing: {mp}')
    m=json.loads(mp.read_text(encoding='utf-8')); obj=objective(m)
    r={'run':rid,'stage':stage,'params':dict(p, gap=gap),'requested_params':p,'actual_gap':gap,'cheb_lengths':cheb,'butter_lengths':butter,'metrics':m,'objective':obj,'elapsed_s':elapsed,'tag':tag,'log':str(log)}
    with HISTORY_PATH.open('a',encoding='utf-8') as fh: fh.write(json.dumps(r)+'\n')
    print(f'RESULT {rid:04d}: objective={obj:.3f} | S21@9={m["s21_9GHz_dB"]:.2f} dB | S21@10={m["s21_10GHz_dB"]:.2f} dB | max14-18={m["max_s21_14_18_dB"]:.2f} dB | max20-100={m["max_s21_20_100_dB"]:.2f} dB')
    return r

def key(p): return tuple(round(p[k],6) for k in BASE)
def load():
    seen=set(); rec=[]
    if HISTORY_PATH.exists():
      for line in HISTORY_PATH.read_text(encoding='utf-8').splitlines():
        if line.strip():
          r=json.loads(line); rec.append(r); seen.add(key(r['params']))
    return seen,rec
def evaluate(p,rid,stage,seen,rec):
    k=key(p)
    if k in seen:
      return next(r for r in rec if key(r['params'])==k)
    r=run_candidate(p,rid,stage); seen.add(k); rec.append(r); return r

def main():
    CAMPAIGN_DIR.mkdir(parents=True,exist_ok=True); LOG_DIR.mkdir(parents=True,exist_ok=True); make_template()
    seen,rec=load(); best=min(rec,key=lambda r:r['objective']) if rec else None; rid=max([r['run'] for r in rec],default=0)+1
    current=dict(best['params'] if best else BASE)
    print('='*92); print('TARGETED B3 EM OPTIMIZER'); print('Goal: cutoff ~10 GHz, suppress 14–18 GHz resonance, drive 20–100 GHz toward -40 dB'); print(f'Completed runs: {len(rec)}')
    r=evaluate(current,rid,'BASELINE',seen,rec); rid+=1; current=dict(r['params']); cur_obj=r['objective']; best=r if best is None or r['objective']<best['objective'] else best
    for st in STAGES:
      name=st['name']; steps=st['steps']; print('\n'+'#'*92); print(f'STAGE {name}')
      for passno in range(1,3):
        improved=False; print(f'\n{name} PASS {passno} | objective={cur_obj:.3f}')
        for var in BASE:
          for val in [clamp(var,current[var]-steps[var]),clamp(var,current[var]+steps[var])]:
            trial=dict(current); trial[var]=val; rr=evaluate(trial,rid,name,seen,rec); rid+=1
            if rr['objective']+1e-9<cur_obj:
              current=dict(rr['params']); cur_obj=rr['objective']; improved=True; best=rr if best is None or rr['objective']<best['objective'] else best; print(f'KEEP {var}={current[var]:.6f} -> objective={cur_obj:.3f}')
        if not improved: break
    if best:
      out={'objective':best['objective'],'params':best['params'],'cheb_lengths':best['cheb_lengths'],'butter_lengths':best['butter_lengths'],'metrics':best['metrics'],'tag':best['tag']}
      BEST_PATH.write_text(json.dumps(out,indent=2),encoding='utf-8')
      print('\n'+'='*92); print('OPTIMIZATION COMPLETE'); print(json.dumps(out,indent=2)); print(f'Best: {BEST_PATH}'); print(f'History: {HISTORY_PATH}')
if __name__=='__main__': main()
