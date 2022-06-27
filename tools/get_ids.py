import argparse
import pandas

def main():
    parser = argparse.ArgumentParser(description="Performe feature selection on dataset.")
    parser.add_argument("dataset", help="Path to CSV file.")
    parser.add_argument("--attack", action="store_true", help="Filter IDs that are attacked.")
    args = parser.parse_args()
    dataframe = pandas.read_csv(parser.parse_args().dataset)
    if args.attack:
        ids = dataframe.loc[dataframe.iloc[:, -2] == "Attack"].iloc[:, 1].unique()
        ids.sort()
        print(f"There are {len(ids)} unique IDs targeted for attacks: {ids}")
    else:
        ids = dataframe.iloc[:, 1].unique()
        ids.sort()
        print(f"There are {len(ids)} unique IDs: {ids}")

if __name__ == "__main__":
    main()