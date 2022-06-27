import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Generate plot from CSV data.")
    parser.add_argument("dataset", help="Path to CSV file.")
    args = parser.parse_args()
    Path("graphs").mkdir(exist_ok=True)

    data = pd.read_csv(args.dataset)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.set_size_inches(15, 5)

    line1, = ax1.plot(data["N"], data["False negative rate"], label="False negative rate")
    line2, = ax1.plot(data["N"], data["Error rate"], label="Error rate")
    line3, = ax1.plot(data["N"], data["Precision"], label="Precision")
    line4, = ax1.plot(data["N"], data["Recall"], label="Recall")
    line5, = ax1.plot(data["N"], data["F1-score"], label="F1-score")
    line6, = ax2.plot(data["N"], data["Windows"], label="Number of training feature sets", color="black")


    ax1.set_xlabel("Number of IDs")
    ax1.set_ylabel("Performance (%)")
    ax2.set_ylabel("Number of training feature sets")

    plt.xticks(data["N"])
    plt.legend(handles = [line1, line2, line3, line4, line5, line6], loc = "center right")
    plt.savefig("graphs/varying_id_performance.png")

if __name__ == "__main__":
    main()
