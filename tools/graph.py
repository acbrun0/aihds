import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Generate plot from CSV data.")
    parser.add_argument("dataset", nargs="*", help="Path to CSV file.")
    parser.add_argument("--reference", help="Path to CSV file.")
    parser.add_argument("--compare", nargs="*", help="Path to CSV file.")
    args = parser.parse_args()
    Path("graphs").mkdir(exist_ok=True)

    if args.reference is not None and args.compare is not None:
        normal = pd.read_csv(args.reference, usecols=[0, 1], names=["Entropy", "Hamming distance"])
        for path in args.compare:
            compare = pd.read_csv(path, usecols=[0, 1, 2], names=["Entropy", "Hamming distance", "Label"])
            ax = plt.figure().add_subplot()
            ax.scatter(normal["Entropy"], normal["Hamming distance"], label=args.reference)
            ax.scatter(compare["Entropy"], compare["Hamming distance"], c=np.where(compare["Label"], "r", "g"), label=path)
            plt.xlabel("Entropy")
            plt.ylabel("Hamming distance")
            plt.legend(loc="best")
            plt.savefig(f"graphs/{Path(path).parent.name}.png")
    else:
        for path in args.dataset:
            Path(f"graphs/{Path(path).stem}").mkdir(exist_ok=True)
            dataset = pd.read_csv(path)
            for col in dataset:
                if col != "Label":
                    figure(figsize=(15, 5))
                    plt.xlabel("Window")
                    plt.ylabel(col)
                    plt.scatter(np.arange(0, len(dataset[col])), dataset[col], c=np.where(dataset["Label"], "r", "g"))
                    plt.savefig(f"graphs/{Path(path).stem}/{col}.png")


if __name__ == "__main__":
    main()