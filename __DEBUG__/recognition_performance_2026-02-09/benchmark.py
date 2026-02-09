import subprocess
import re
import json
import time
import os
from pathlib import Path

# Configuration
CONFIGS = [
    # {"name": "baseline_nw0_bs16", "workers": 0, "pin_memory": "false", "batch_size": 16}, # Already run
    {"name": "mp_nw4_bs16", "workers": 4, "pin_memory": "true", "batch_size": 16},
    {"name": "mp_nw8_bs16", "workers": 8, "pin_memory": "true", "batch_size": 16},
    {"name": "mp_nw4_bs64", "workers": 4, "pin_memory": "true", "batch_size": 64},
    # {"name": "mp_nw4_bs128", "workers": 4, "pin_memory": "true", "batch_size": 128},
]

BASE_CMD = [
    "uv", "run", "python", "scripts/runners/train.py",
    "domain=recognition",
    "experiment=rec_baseline_official",
    "trainer.max_epochs=1",
    "trainer.limit_train_batches=30",
    "trainer.limit_val_batches=0",
    "hydra.run.dir=__DEBUG__/recognition_performance_2026-02-09/logs/{name}"
]

RESULTS_FILE = Path("__DEBUG__/recognition_performance_2026-02-09/benchmark_results.json")

def parse_speed(log_output):
    matches = re.findall(r"(\d+\.\d+)it/s", log_output)
    if matches:
        return float(matches[-1])
    return 0.0

def run_benchmark():
    results = []

    # Add known baseline result
    results.append({
        "name": "baseline_nw0_bs16",
        "workers": 0, "pin_memory": False, "batch_size": 16,
        "speed": 7.46,
        "status": "SUCCESS",
        "duration": 0
    })

    for cfg in CONFIGS:
        print(f"Running {cfg['name']}...")
        config_name = cfg['name']

        cmd = BASE_CMD[:]

        # Override DATALOADERS config directly
        cmd.append(f"dataloaders.train_dataloader.num_workers={cfg['workers']}")
        cmd.append(f"dataloaders.val_dataloader.num_workers={cfg['workers']}")

        cmd.append(f"dataloaders.train_dataloader.pin_memory={cfg['pin_memory']}")
        cmd.append(f"dataloaders.val_dataloader.pin_memory={cfg['pin_memory']}")

        # Use + because batch_size is not in the default struct for dataloaders
        cmd.append(f"+dataloaders.train_dataloader.batch_size={cfg['batch_size']}")
        cmd.append(f"+dataloaders.val_dataloader.batch_size={cfg['batch_size']}")

        # Also set data.batch_size just in case something else uses it
        cmd.append(f"data.batch_size={cfg['batch_size']}")

        # Fix dynamic hydra dir
        fixed_cmd = []
        for arg in cmd:
            if "{name}" in arg:
                fixed_cmd.append(arg.format(name=config_name))
            else:
                fixed_cmd.append(arg)

        start_time = time.time()
        output = ""
        status = "UNKNOWN"
        speed = 0.0

        try:
            print(f"Executing: {' '.join(fixed_cmd)}")
            result = subprocess.run(
                fixed_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout + result.stderr
            speed = parse_speed(output)
            status = "SUCCESS"
        except subprocess.CalledProcessError as e:
            output = e.stdout + e.stderr
            speed = 0.0
            status = "FAILED"
            print(f"Failed: {e}")

        duration = time.time() - start_time

        res_entry = {
            "name": config_name,
            "workers": cfg["workers"],
            "pin_memory": cfg["pin_memory"],
            "batch_size": cfg["batch_size"],
            "speed": speed,
            "status": status,
            "duration": duration
        }
        results.append(res_entry)
        print(f"Result: {res_entry}")

        log_file = Path(f"__DEBUG__/recognition_performance_2026-02-09/logs/{config_name}.log")
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(output)

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print("Benchmark complete. Results saved.")

if __name__ == "__main__":
    run_benchmark()
