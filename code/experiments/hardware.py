"""Record the processor, memory and software versions of the machine that runs an experiment."""
from __future__ import annotations

import os
import platform
import subprocess
import sys


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def describe() -> dict:
    cpu, ram = platform.processor(), ""
    if sys.platform.startswith("win"):
        cpu = _cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]) or cpu
        ram = _cmd(["powershell", "-NoProfile", "-Command",
                    "[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"])
    elif sys.platform == "darwin":
        cpu = _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or cpu
        ram = str(round(int(_cmd(["sysctl", "-n", "hw.memsize"]) or 0) / 2**30, 1))
    else:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                ram = str(round(int(line.split()[1]) / 2**20, 1)); break
    import numpy, scipy, pandas
    try:
        import pymoo; pv = pymoo.__version__
    except Exception:
        pv = ""
    return {"cpu": cpu, "logical_cores": os.cpu_count(), "ram_gb": ram, "os": platform.platform(),
            "python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "pandas": pandas.__version__, "pymoo": pv}


def record(path):
    info = describe()
    with open(path, "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    print("machine:", info["cpu"], "|", info["logical_cores"], "logical cores |", info["ram_gb"], "GB")
