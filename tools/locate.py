import argparse
import pandas

def main():
    parser = argparse.ArgumentParser(description="Locate values on IEEE challenge dataset.")
    parser.add_argument("dataset", nargs="*", help="Path to CSV file.")
    args = parser.parse_args()

    for path in args.dataset:
        dataframe = pandas.read_csv(path)
        dataframe = dataframe.loc[dataframe["label"] == "Attack"]
        last_attack = dataframe.iloc[0][-1]
        last_timestamp = dataframe.iloc[0]["timestamp"]
        start = last_timestamp
        end = 0

        for _, row in dataframe.iterrows():
            if row["class"] != last_attack:
                end = last_timestamp
                print(f"{last_attack} occurs from {start} to {end}")
                start = row["timestamp"]
                last_attack = row["class"]
            last_timestamp = row["timestamp"]

        

if __name__ == "__main__":
    main()