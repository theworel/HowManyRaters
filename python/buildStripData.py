import csv
import statistics
from pathlib import Path

# Reshape the wide per-class perplexity matrices written by buildPerpsCSV30.py
# into long-format data for a LaTeX (pgfplots) strip plot.
#
# Input:  output30/perps{Good,Bad,Broken,Corrupt}.csv
#         wide matrix, first column = measured model, remaining byXXX columns
#         = perplexity as measured by each measuring model.
# Output: output30/stripdata.csv     -> class,xbase,perp   (one row per point)
#         output30/stripsummary.csv  -> class,xbase,mean,median,n

BASE = Path("/storage/praha1/home/tondaotm/npfl087/npfl087-otmara/resources/output30")

# class label -> x position on the strip plot (controls left-to-right order)
CLASSES = [("Good", 1), ("Bad", 2), ("Broken", 3), ("Corrupt", 4)]


def read_values(csv_path):
    """All numeric perplexity cells in a wide perps*.csv, flattened."""
    vals = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        value_cols = [c for c in reader.fieldnames if c != "measuringModelName"]
        for row in reader:
            for col in value_cols:
                cell = (row.get(col) or "").strip()
                if cell == "":
                    continue
                try:
                    vals.append(float(cell))
                except ValueError:
                    continue
    return vals


def main():
    data_path = BASE / "stripdata.csv"
    summary_path = BASE / "stripsummary.csv"

    with open(data_path, "w", newline="") as df, \
         open(summary_path, "w", newline="") as sf:
        dw = csv.writer(df)
        sw = csv.writer(sf)
        dw.writerow(["class", "xbase", "perp"])
        sw.writerow(["class", "xbase", "mean", "median", "n"])

        for label, xbase in CLASSES:
            csv_path = BASE / f"perps{label}.csv"
            if not csv_path.is_file():
                print("missing", csv_path)
                continue
            vals = read_values(csv_path)
            if not vals:
                print("no values", csv_path)
                continue
            for v in vals:
                dw.writerow([label.lower(), xbase, v])
            sw.writerow([
                label.lower(), xbase,
                statistics.mean(vals), statistics.median(vals), len(vals),
            ])
            print(f"{label}: {len(vals)} points")

    print("wrote", data_path)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
