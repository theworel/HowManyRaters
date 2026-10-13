import csv
import re
from pathlib import Path

MODELS = [
    "eu", "aya", "arnir", "qwen", "ayabig", "dolphin",
    "liquidai", "ibm", "gemma", "gptoss", "darwin", "cfourai"
]

CLASSES = ["bad", "good", "broken", "corrupt", "corruptTriple", "candidateCorrupt"]

def parse_mean(filepath):
    try:
        lines = filepath.read_text().splitlines()
        if len(lines) < 2:
            return None
        m = re.match(r"mean\s+([\d.eE+\-]+)", lines[1])
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None

def main():
    base = Path("/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/resources/output30/perps")

    # data[cls][measured_model][measuring_model] = float value
    data = {cls: {m: {} for m in MODELS} for cls in CLASSES}

    for measuring_model in MODELS:
        by_dir = base / f"by{measuring_model}"
        if not by_dir.is_dir():
            print("not dir - by")
            continue
        for cls in CLASSES:
            cls_dir = by_dir / cls
            if not cls_dir.is_dir():
                print("not dir - class")
                continue
            for measured_model in MODELS:
                filepath = cls_dir / f"of{measured_model}.txt"
                if not filepath.is_file():
                    print("no file", filepath)
                    continue
                value = parse_mean(filepath)
                if value is not None:
                    print("value", value)
                    data[cls][measured_model][measuring_model] = value
                else:
                    print("val none", filepath)

    out_dir = Path("/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/resources/output30")
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = ["measuringModelName"] + [f"by{m}" for m in MODELS]

    for cls in CLASSES:
        out_path = out_dir / f"perps{cls.capitalize()}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for measured_model in MODELS:
                row = {"measuringModelName": measured_model}
                for measuring_model in MODELS:
                    val = data[cls][measured_model].get(measuring_model)
                    if val is not None:
                        row[f"by{measuring_model}"] = val
                writer.writerow(row)
        
        print(f"Written {out_path}")

if __name__ == "__main__":
    main()
