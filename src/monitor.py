import psutil
import time
import os

# GPU (NVIDIA)
try:
    from pynvml import *
    nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

def clear():
    os.system("clear")

def get_gpu_info():
    if not GPU_AVAILABLE:
        return None

    try:
        handle = nvmlDeviceGetHandleByIndex(0)

        util = nvmlDeviceGetUtilizationRates(handle)
        mem = nvmlDeviceGetMemoryInfo(handle)

        return {
            "gpu": util.gpu,
            "vram": (mem.used / mem.total) * 100,
            "vram_used": mem.used / (1024**3),
            "vram_total": mem.total / (1024**3)
        }
    except:
        return None

def get_disks():
    disks = []
    seen_devices = set()

    for part in psutil.disk_partitions(all=False):
        # Evita listar o mesmo device físico montado em vários pontos
        # (comum com binds, snaps, etc.)
        if part.device in seen_devices:
            continue

        try:
            usage = psutil.disk_usage(part.mountpoint)
            seen_devices.add(part.device)
            disks.append({
                "device": part.device,
                "mount": part.mountpoint,
                "fstype": part.fstype,
                "percent": usage.percent,
                "used_gb": usage.used / (1024**3),
                "free_gb": usage.free / (1024**3),
                "total_gb": usage.total / (1024**3),
            })
        except (PermissionError, OSError):
            continue
    return disks

def bar(percent, width=30):
    filled = int(width * percent / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

while True:
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    disks = get_disks()
    gpu = get_gpu_info()

    clear()

    print("======== MONITOR DE RECURSOS ========\n")

    # CPU
    print(f"CPU: {cpu:.1f}%")

    # RAM
    print(f"RAM: {ram.percent:.1f}% "
          f"({ram.used / (1024**3):.1f}GB / {ram.total / (1024**3):.1f}GB)")

    print("\n--- DISCOS ---")
    for d in disks:
        print(f"\n{d['mount']}  ({d['device']}, {d['fstype']})")
        print(f"  {bar(d['percent'])} {d['percent']:.1f}%")
        print(f"  Usado: {d['used_gb']:.1f}GB | "
              f"Livre: {d['free_gb']:.1f}GB | "
              f"Total: {d['total_gb']:.1f}GB")

    print("\n--- GPU ---")
    if gpu:
        print(f"GPU: {gpu['gpu']:.1f}%")
        print(f"VRAM: {gpu['vram']:.1f}% "
              f"({gpu['vram_used']:.2f}GB / {gpu['vram_total']:.2f}GB)")
    else:
        print("GPU: indisponível")

    print("\n(Atualizando a cada 1s...)")

    time.sleep(1)