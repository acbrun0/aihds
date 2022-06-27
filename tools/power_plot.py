import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure

def main():
    parser = argparse.ArgumentParser(description="Generate plot from CSV data.")
    parser.add_argument("dataset", help="Path to CSV file.")
    args = parser.parse_args()
    Path("graphs").mkdir(exist_ok=True)
    
    dataframe = pd.read_csv(args.dataset)
    figure(figsize=(20, 5))
    plt.plot(np.arange(0, dataframe.shape[0]), dataframe["Current(mA)"])
    plt.savefig("graphs/powerdraw.png")

if __name__ == "__main__":
    main()