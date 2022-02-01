import os
import pandas
import concurrent.futures
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from argparse import ArgumentParser

def convert(path: str):

	def write_file(dataset: pandas.DataFrame, path: str):
		with open(path, "w") as file:
			for _, row in dataset.iterrows():
				file.write(f"{row['label'] + ' ' if row['label'] != '' else ''} "
						   f"0:{row['id']} "
						   f"1:{row['dlc']} "
						   f"{'2:' + str(row['data0']) if not pandas.isnull(row['data0']) else ''} "
						   f"{'3:' + str(row['data1']) if not pandas.isnull(row['data1']) else ''} "
						   f"{'4:' + str(row['data2']) if not pandas.isnull(row['data2']) else ''} "
						   f"{'5:' + str(row['data3']) if not pandas.isnull(row['data3']) else ''} "
						   f"{'6:' + str(row['data4']) if not pandas.isnull(row['data4']) else ''} "
						   f"{'7:' + str(row['data5']) if not pandas.isnull(row['data5']) else ''} "
						   f"{'8:' + str(row['data6']) if not pandas.isnull(row['data6']) else ''} "
						   f"{'9:' + str(row['data7']) if not pandas.isnull(row['data7']) else ''}\n"
						  )
		return path

	def set_label(label: str):
		if label == 'T': return "+1"
		else: return '-1'

	def structure_row(row):
		dlc = int(row[2])
		if dlc < 8:
			# Set label
			row[-1] = row[3 + dlc]
			# Set NaN for missing bytes.
			for i in range(3 + dlc, 10):
				row[i] = np.NaN
		return row

	dataset_path = path.split(os.sep)[-2:]
	os.makedirs(f"datasets/libsvm/{dataset_path[0]}/{dataset_path[1].split('.')[0]}", exist_ok = True)
	df = pandas.read_csv(path, names = ["timestamp", "id", "dlc", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7", "label"])

	print("Loaded dataset.")

	df = df.apply(structure_row, axis = 1)
	df["label"] = df["label"].apply(set_label)
	df[["id", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7"]] = df[["id", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7"]].applymap(lambda byte: int(byte, 16) if not pandas.isnull(byte) else byte)
	df[["id", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7"]] = StandardScaler().fit_transform(df[["id", "data0", "data1", "data2", "data3", "data4", "data5", "data6", "data7"]])

	print("Finished processing.")

	train, test = train_test_split(df, train_size = 0.6, shuffle = False)

	with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:
		future_to_file = {executor.submit(write_file, dataset, f"datasets/libsvm/{dataset_path[0]}/{dataset_path[1].split('.')[0]}/{name}.txt"): (dataset, name) for dataset, name in [(train, "train"), (test, "test")]}
		print("Writing to files.")
		for future in concurrent.futures.as_completed(future_to_file):
			print(f"Finished writing to {future.result()}")

if __name__ == "__main__":
	parser = ArgumentParser(description = "Convert CAN dataset to libsvm compatible format.")
	parser.add_argument("path", help = "Path to the dataset.")
	args = parser.parse_args()
	convert(args.path)