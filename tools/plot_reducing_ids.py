import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

def main():
    parser = argparse.ArgumentParser(description="Generate plot from CSV data.")
    parser.add_argument("dataset", nargs="*", help="Path to CSV file.")
    args = parser.parse_args()
    Path("graphs").mkdir(exist_ok=True)

    entropy = []
    hamming = []
    for path in args.dataset:
        dataframe = pd.read_csv(path)
        entropy.append(dataframe["Entropy"])
        hamming.append(dataframe["HammingDistBytes"])
    labels = map(lambda p: Path(p).stem.split('_')[-1], args.dataset)
    
    figure(figsize=(20, 5))
    for e, label in zip(entropy, labels):
        plt.plot(np.arange(0, len(e)), e, label=label)
    plt.legend()
    plt.savefig("graphs/entropy.png")


if __name__ == "__main__":
    main()