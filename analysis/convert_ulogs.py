import os
import argparse
import pandas as pd
from pyulog import ULog

def safe_get(ulog: ULog, name: str):
    try:
        return ulog.get_dataset(name).data
    except (KeyError, IndexError):
        return None

def convert_ulogs_to_csv(input_dir: str):
    # 1. Create the csv base directory inside the input directory
    output_base = os.path.join(input_dir, "csv")
    os.makedirs(output_base, exist_ok=True)

    # 2. Loop through each file in the input directory
    for filename in os.listdir(input_dir):
        if not filename.endswith(".ulg"):
            continue

        base_name = filename[:-4]  # strip ".ulg"
        ulog_path = os.path.join(input_dir, filename)
        output_dir = os.path.join(output_base, f"{base_name}_csv")
        os.makedirs(output_dir, exist_ok=True)

        # 3. Load the ULog
        ulog = ULog(ulog_path)

        # 4. Save each dataset as CSV
        for dataset in ulog.data_list:
            data = safe_get(ulog, dataset.name)
            if data is None:
                continue
            df = pd.DataFrame(data)
            csv_filename = f"{dataset.name}.csv"
            csv_path = os.path.join(output_dir, csv_filename)
            df.to_csv(csv_path, index=False)

        print(f"Converted {filename} → {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all ULog files in a directory to CSVs")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the directory containing .ulg files"
    )
    args = parser.parse_args()
    convert_ulogs_to_csv(args.input_dir)
