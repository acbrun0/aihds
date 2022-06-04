import argparse
import pandas

def main():
    parser = argparse.ArgumentParser(description="Performe feature selection on dataset.")
    parser.add_argument("dataset", help="Path to CSV file.")
    ids = pandas.read_csv(parser.parse_args().dataset).iloc[:, 1].unique()
    print(f"There are {len(ids)} unique IDs: {ids}")

if __name__ == "__main__":
    main()