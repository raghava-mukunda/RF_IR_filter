from pathlib import Path

RUN_DIR = Path(
    r"C:\Users\Raghava.m\Desktop\cryogenic_filter_design"
    r"\FINAL_OPENEMS_RIGOROUS\rf_final_300k"
)

for name in [
    "port_ut_1",
    "port_ut_2",
    "port_it_1",
    "port_it_2",
]:
    path = RUN_DIR / name

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    with open(path, "r", errors="replace") as f:
        for i in range(15):
            line = f.readline()
            if not line:
                break

            print(f"{i+1:03d}: {repr(line)}")