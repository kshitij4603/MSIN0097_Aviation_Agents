import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS = [
    BASE_DIR / 'task2_eda.py',
    BASE_DIR / 'task2_basic_eda.py',
    BASE_DIR / 'task2_advanced_stats.py',
    BASE_DIR / 'task3_baseline_model.py',
    BASE_DIR / 'task4_optimized_model.py',
    BASE_DIR / 'task5_model_audit.py',
]

if __name__ == '__main__':
    print(f'Running pipeline from {BASE_DIR}')
    for script in SCRIPTS:
        if not script.exists():
            raise FileNotFoundError(f'Missing script: {script}')
        print(f'Executing: {script}')
        cmd = [sys.executable, str(script)]
        res = subprocess.run(cmd, cwd=BASE_DIR.parent)
        if res.returncode != 0:
            raise RuntimeError(f'Script failed: {script} with code {res.returncode}')
    print('Pipeline completed successfully.')
