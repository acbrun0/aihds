import argparse
import can

def main():
    parser = argparse.ArgumentParser(description="Calculates the average number of packets per second in the CSV file.")
    parser.add_argument("dataset", help="Path to CSV file.")
    args = parser.parse_args()

    timestamps = list(map(lambda msg: msg.timestamp, list(can.BLFReader(args.dataset))))

    diffs = []
    for i in range(0, len(timestamps) - 1):
        diffs.append(timestamps[i + 1] - timestamps[i])
    
    print(f"{1 / (sum(diffs) / len(diffs)):.3f} packets/s")


if __name__ == "__main__":
    main()