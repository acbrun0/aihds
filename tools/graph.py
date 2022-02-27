import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate plot from CSV data.")
    parser.add_argument("reference", help="Path to CSV file.")
    parser.add_argument("compare", nargs="+", help="Path to CSV file.")
    args = parser.parse_args()
    Path("graphs").mkdir(exist_ok=True)

    normal = pd.read_csv(args.reference, header=None, usecols=[0, 1], names=["General entropy", "Entropy by ID"])
    for path in args.compare:
        compare = pd.read_csv(path, header=None, usecols=[0, 1, 2], names=["General entropy", "Entropy by ID", "Label"])
        ax = plt.figure().add_subplot()
        ax.scatter(normal["Entropy by ID"], normal["General entropy"], label=args.reference)
        plt.xlabel("Entropy by ID")
        plt.ylabel("General entropy")
        ax.scatter(compare["Entropy by ID"], compare["General entropy"], c=np.where(compare["Label"], "r", "g"), label=path)
        plt.legend(loc="best")
        plt.savefig(f"graphs/{Path(path).parent.name}.png")

if __name__ == "__main__":
    main()