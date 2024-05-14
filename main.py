import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
import os
from fuzzywuzzy import process
import pandas as pd
from collections import defaultdict
from pyarrow import parquet as pq
from tables import NaturalNameWarning
import warnings
import json
import traceback

# Suppress NaturalNameWarning
warnings.filterwarnings("ignore", category=NaturalNameWarning)


def download_data(data_folder="data"):
    """""
    Download the data from the NYC Open Data website.

    Args:
        data_folder (str): The folder where the data will be downloaded. Defaults to "data".

    Returns:
        None
    """ ""
    # run data/nyt.sh script to download the data
    os.system(f"sh {data_folder}/nyt.sh")


def create_column_mapping(reference_cols, current_cols):
    """
    Create a mapping from current_cols to reference_cols based on string similarity,
    ensuring that no duplicate mappings are created.
    """
    used_names = set()  # To track names already used in the mapping
    column_mapping = {}
    similarity_scores = defaultdict(list)

    # First, gather all potential mappings with their scores
    for current in current_cols:
        for reference in reference_cols:
            score = process.extractOne(current, [reference])[1]
            similarity_scores[current].append((score, reference))

    # Sort potential mappings by score
    for current, scores in similarity_scores.items():
        sorted_scores = sorted(scores, key=lambda x: -x[0])  # Sort by score descending

        # Find the highest-score mapping that hasn't been used yet
        for score, reference in sorted_scores:
            if reference not in used_names:
                column_mapping[current] = reference
                used_names.add(reference)
                break
        else:
            # If all scores are used, keep the original
            column_mapping[current] = current

    return column_mapping
# Define a function to limit string columns to 300 characters
def limit_string_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.slice(0, 300)
    return df

def read_and_process_file(file, reference_cols, base_path="data/", temp_dir="temp/"):
    """
    Read a file, apply column renaming, and save the DataFrame to a temporary location.
    Return the path to the temporary file.
    """
    # Ensure the temporary directory exists
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Read the file with Dask
    df = dd.read_csv(os.path.join(base_path, file), dtype="object")
    # Apply the function to each partition
    df = df.map_partitions(limit_string_columns)
    # Create mapping and apply renaming
    current_cols = df.columns
    mapping = create_column_mapping(reference_cols, current_cols)
    df = df.rename(columns=mapping)

    # Path for the temporary file
    temp_file = os.path.join(temp_dir, os.path.basename(file))

    # Save to Parquet to maintain data types correctly
    df.to_parquet(temp_file)

    return temp_file


def process_all_files(directory="data/", reference_file="2024.csv", temp_dir="temp/"):
    '''client = Client(
        n_workers=12,
        threads_per_worker=1,
        memory_limit="3GB",
        local_directory="/tmp",
    )'''
    client = Client()
    reference_path = os.path.join(directory, reference_file)
    reference_df = pd.read_csv(reference_path, nrows=0)
    reference_cols = list(reference_df.columns)

    files = [
        f for f in os.listdir(directory) if f.endswith(".csv") and f != reference_file
    ]
    temp_files = [
        read_and_process_file(f, reference_cols, directory, temp_dir) for f in files
    ]

    # Read all temporary Parquet files into Dask DataFrames and concatenate
    dataframes = [dd.read_parquet(file) for file in temp_files]
    if dataframes:
        combined_df = dd.concat(dataframes, axis=0, ignore_index=True)
        combined_df.to_parquet(
            "data/processed/combined_parking_violations.parquet", write_index=False
        )
    else:
        print("No dataframes to concatenate.")

    client.close()

def parquet_to_hdf5():
    parquet_directory = "data/processed/combined_parking_violations.parquet"
    hdf5_path = "data/processed/combined_parking_violations.h5"
    log_file = "data/processed/progress_log.txt"
    min_itemsize_file = "data/processed/min_itemsize.json"

    batch_size = 1_000_000

    parquet_files = [f for f in os.listdir(parquet_directory) if f.endswith(".parquet")]

    # Initialize min_itemsize dictionary 
    '''default_min_itemsize = {
        "Issuing Agency": 10,
        "Sub Division": 10,
        "House Number": 20,
        "Violation Location": 10,
        "Violation County": 10,
        "Violation Description": 80,
        "Vehicle Expiration Date": 30,
        "Date First Observed": 30,
        "Issue Date": 30,
        "Issuer Command": 10,
        "Feet From Curb": 10,
        "Street Name": 300,
        "Violation Post Code": 300,
        "Plate ID": 20,
        "No Standing or Stopping Violation" : 40,
    }'''
    default_min_itemsize = {
        "Issuing Agency": 10,
        "Sub Division": 10,
        "House Number": 20,
        "Violation Location": 10,
        "Violation County": 10,
        "Violation Description": 80,
        "Vehicle Expiration Date": 30,
        "Date First Observed": 30,
        "Issue Date": 30,
        "Issuer Command": 10,
        "Feet From Curb": 10,
        "Street Name": 300,
        "Violation Post Code": 300,
        "Plate ID": 20,
        "No Standing or Stopping Violation" : 40,
    }
    # Load progress from log file if it exists
    if os.path.exists(log_file):
        with open(log_file, "r") as log:
            processed_files = set(log.read().splitlines())
    else:
        processed_files = set()

    # Load min_itemsize from file if it exists
    if os.path.exists(min_itemsize_file):
        with open(min_itemsize_file, "r") as f:
            min_itemsize = json.load(f)
            print("Loaded min_itemsize from file.")
            print(min_itemsize)
    else:
        min_itemsize = default_min_itemsize

    length = len(parquet_files)
    for i, filename in enumerate(parquet_files):
        if filename in processed_files:
            continue  # Skip already processed files

        file_path = os.path.join(parquet_directory, filename)
        parquet_file = pq.ParquetFile(file_path)
        # Print to output
        print(f"Processing file: {filename}", flush=True)
        print(f"Progress: {i + 1}/{length}; {((i+1)/length)*100}%", flush=True)
        print("-" * 50, flush=True)

        try:
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                df = batch.to_pandas()
                print(df.columns)
                # Limit string columns to 300 characters
                #df = df.map(lambda x: x[:300] if isinstance(x, str) else x)

                # Update min_itemsize for each string column
                for col, dtype in df.dtypes.items():
                    if pd.api.types.is_string_dtype(dtype) or dtype == "string":
                        df[col] = df[col].astype("object")
                        max_length = (
                            df[col].str.len().max() or 0
                        )  # Avoid NaN by using or 0
                        max_length = min(max_length, 300)  # Cap the length at 300

                        # Ensure min_itemsize is updated correctly and does not reset
                        if col not in min_itemsize or min_itemsize[col] < max_length:
                            min_itemsize[col] = max_length

                # Remove NaN values from min_itemsize
                min_itemsize = {k: v for k, v in min_itemsize.items() if pd.notnull(v)}

                # Append to HDF5 with maximal compression and current min_itemsize
                df.to_hdf(
                    hdf5_path,
                    key="data",
                    mode="a",
                    format="table",
                    data_columns=True,
                    append=True,
                    complevel=9,
                    complib="blosc:blosclz",
                    min_itemsize=min_itemsize,
                )

            # Update progress log file
            with open(log_file, "a") as log:
                log.write(filename + "\n")

            # Save updated min_itemsize to file
            with open(min_itemsize_file, "w") as f:
                json.dump(min_itemsize, f)

        except Exception as e:
            print(f"Error processing file: {filename}", flush=True)
            print(traceback.format_exc(), flush=True)
            break  # Stop processing further files in case of an error

def compare_file_sizes(hdf5_path, parquet_directory):
    """
    Compare the size of a HDF5 file to the combined size of the Parquet files.
    """
    parquet_files = [f for f in os.listdir(parquet_directory) if f.endswith(".parquet")]

    # Calculate total size of Parquet files
    parquet_size = sum(os.path.getsize(os.path.join(parquet_directory, f)) for f in parquet_files)

    # Get size of HDF5 file
    hdf5_size = os.path.getsize(hdf5_path)

    # Return sizes in bytes and gigabytes, as well as the ratio of HDF5 to Parquet size
    return {
        "parquet_size_bytes": parquet_size,
        "hdf5_size_bytes": hdf5_size,
        "parquet_size_gb": parquet_size / 1e9,
        "hdf5_size_gb": hdf5_size / 1e9,
        "ratio": hdf5_size / parquet_size,
    }

def run_task_1():
    process_all_files()
    parquet_to_hdf5()
    compare_file_sizes("data/processed/combined_parking_violations.h5", "data/processed/combined_parking_violations.parquet")


if __name__ == "__main__":
    # download_data()
    #process_all_files(reference_file='parking_violations_2024.csv')
    parquet_to_hdf5()
