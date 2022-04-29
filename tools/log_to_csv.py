import argparse
import re
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert CANDump format to CSV.")
    parser.add_argument("dataset", help="Path to CSV file.")
    args = parser.parse_args()

    with open(args.dataset, "r") as file:
        content = file.read()
        content = re.sub("\(", '', content, flags=re.M)
        content = re.sub("\) slcan0 ", ',', content, flags=re.M)
        content = re.sub(f"#(?=[0-9A-Z]{{16}})", ",8,", content, flags=re.M)
        content = re.sub(f"#(?=[0-9A-Z]{{14}})", ",7,", content, flags=re.M)
        content = re.sub(f"#(?=[0-9A-Z]{{12}})", ",6,", content, flags=re.M)
        content = re.sub(f"#(?=[0-9A-Z]{{10}})", ",5,", content, flags=re.M)
        content = re.sub(f"#(?=[0-9A-Z]{{8}})", ",4,", content, flags=re.M)
        content = re.sub(f"#(?=[0-9A-Z]{{6}})", ",3,", content, flags=re.M)
        content = re.sub(f"#(?=[0-9A-Z]{{4}})", ",2,", content, flags=re.M)
        content = re.sub(f"#(?=[0-9A-Z]{{2}})", ",1,", content, flags=re.M)
        content = re.sub(f"(?<=,[0-9A-Z]{{1}},[0-9A-Z]{{2}})(?=[^,])", ',', content, flags=re.M)
        for _ in range(8):
            content = re.sub(f"(?<=,[0-9A-Z]{{2}},[0-9A-Z]{{2}})(?=[^,\n])", ',', content, flags=re.M)
        content = re.sub('\n', ",Normal\n", content, flags=re.M)
        with open("datasets/replaced.csv", "w") as out:
            out.write("timestamp,id,dlc,data0,data1,data2,data3,data4,data5,data6,data7,label\n" + content)


if __name__ == "__main__":
    main()