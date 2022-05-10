import argparse
import pandas
import matplotlib.pyplot as plt
import seaborn
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Performe feature selection on dataset.")
    parser.add_argument("dataset", help="Path to CSV file.")
    args = parser.parse_args()
    dataset = pandas.read_csv(args.dataset)

    plt.figure(figsize=(15, 10))
    seaborn.heatmap(dataset.corr(), annot=True)
    plt.savefig(f"graphs/{Path(args.dataset).stem}.png")

if __name__ == "__main__":
    main()