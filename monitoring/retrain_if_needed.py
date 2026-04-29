import json
import subprocess
from pathlib import Path

RESULT_PATH = Path("reports/drift_result.json")

def main():
    if not RESULT_PATH.exists():
        print("Nenhum resultado de drift encontrado.")
        return

    data = json.loads(RESULT_PATH.read_text())

    if data.get("retrain_required"):
        print("Drift detectado → iniciando retreino...")

        subprocess.run(["python", "train.py"], check=True)

        print("Retreino concluído.")

    else:
        print("Sem drift relevante. Nenhuma ação necessária.")

if __name__ == "__main__":
    main()