import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def label_to_color_alt(label):
    match label:
        case 0 | 2:
            return 'g'
        case 1 | 3:
            return 'b'

def label_to_color(label, alt):
    match label:
        case 0:
            return 'g'
        case 1:
            return 'b'
        case 2:
            return 'y'
        case 3:
            return 'r'

def main():
    parser = argparse.ArgumentParser(description="Generate plot from CSV data.")
    parser.add_argument("dataset", nargs="*", help="Path to CSV file.")
    parser.add_argument("--threed", action='store_true', help="Make a 3d scatterplot.")
    parser.add_argument("--nolabel", action='store_true', help="Ignore labels.")
    args = parser.parse_args()
    Path("graphs").mkdir(exist_ok=True)
    for path in args.dataset:
        Path(f"graphs/{Path(path).stem}").mkdir(exist_ok=True)
        dataset = pd.read_csv(path)
        if args.threed:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_xlabel("AvgTime")
            ax.set_ylabel("Entropy")
            ax.set_zlabel("HammingDist")
            ax.scatter(dataset["AvgTime"], dataset["Entropy"], dataset["HammingDist"], c = dataset["Label"].apply(label_to_color_alt if args.nolabel else label_to_color).values if "Label" in dataset else "c")
            plt.savefig(f"graphs/{Path(path).stem}.png")
        for col in dataset:
            if col != "Label":
                fig, ax = plt.subplots()
                fig.set_size_inches(20, 5)
                ax.set_xlabel("Window")
                ax.scatter(np.arange(0, len(dataset[col])), dataset[col], c = dataset["Label"].apply(label_to_color_alt if args.nolabel is not None else label_to_color).values if "Label" in dataset else "c")
                if col == "AvgTime":
                    ax.set_ylabel("Average Time")
                    ax.set_title("Average Time Between Packets of the Same ID")
                elif col == "Entropy":
                    ax.set_ylabel(col)
                    ax.set_title("Average Entropy of Packets of the Same ID")
                elif col == "HammingDistBytes":
                    ax.set_ylabel("Hamming Distance")
                    ax.set_title("Average Hamming Distance of Packets of the Same ID")
                if "Label" in dataset:
                    if not args.nolabel:
                        legend_elements = [Line2D([0], [0], marker='o', color='w', label='True positive', markerfacecolor='g', markersize=15),
                                        Line2D([0], [0], marker='o', color='w', label='True negative', markerfacecolor='b', markersize=15),
                                        Line2D([0], [0], marker='o', color='w', label='False positive', markerfacecolor='y', markersize=15),
                                        Line2D([0], [0], marker='o', color='w', label='False negative', markerfacecolor='r', markersize=15)]
                    else:
                        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Alert', markerfacecolor='g', markersize=15),
                                        Line2D([0], [0], marker='o', color='w', label='No alert', markerfacecolor='b', markersize=15)]
                    ax.legend(handles=legend_elements)
                fig.savefig(f"graphs/{Path(path).stem}/{col}.png")


if __name__ == "__main__":
    main()