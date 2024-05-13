import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
import os
import urllib
from fuzzywuzzy import process
import pandas as pd
from collections import defaultdict
import vaex

def download_data(data_folder="data"):
    """""
    Download the data from the NYC Open Data website.

    Args:
        data_folder (str): The folder where the data will be downloaded. Defaults to "data".

    Returns:
        None
    """""
    # run data/nyt.sh script to download the data
    os.system(f"sh {data_folder}/nyt.sh")
    


class Task1:
    def __init__(
        self,
        data_path,
        pq_file="parking_violations_2024.parquet",
        hdf_file="parking_violations_2024.h5",
    ):
        """
        Initializes the Task1 class.

        Args:
            data_path (str): The path to the parking violations data.
            pq_file (str): The name of the parquet file to save the data to. Defaults to "parking_violations_2024.parquet".
            hdf_file (str): The name of the hdf file to save the data to. Defaults to "parking_violations_2024.h5".

        Returns:
            None
        """
        self.data_path = data_path
        self.client = self.set_up_dask_client()
        self.read_data(pq_file, hdf_file)

    def set_up_dask_client(self):
        """
        Sets up a Dask client with the specified configuration.

        Returns:
            client (dask.distributed.Client): The Dask client.
        """
        client = Client(
            n_workers=2,
            threads_per_worker=2,
            memory_limit="6.5GB",
            local_directory="/tmp",
        )
        print(client.scheduler_info()["services"]) 
        return client

    def read_data(
        self,
        pq_file="parking_violations_2024.parquet",
        hdf_file="parking_violations_2024.h5",
    ):
        """
        Reads the parking violations data from the specified data path and saves it to the given parquet and hdf files.

        Args:
            pq_file (str): The name of the parquet file to save the data to. Defaults to "parking_violations_2024.parquet".
            hdf_file (str): The name of the hdf file to save the data to. Defaults to "parking_violations_2024.h5".

        Returns:
            None
        """

        dtype_spec = {
            "summons_number": "int64", "plate_id": "str", "registration_state": "str", "plate_type": "str",
            "violation_code": "int64", "vehicle_body_type": "str", "vehicle_make": "str", "issuing_agency": "str",
            "street_code1": "int64", "street_code2": "int64", "street_code3": "int64",
            "vehicle_expiration_date": "int64", "violation_location": "str", "violation_precinct": "int64",
            "issuer_precinct": "int64", "issuer_code": "int64", "issuer_command": "str", "issuer_squad": "str",
            "violation_time": "str", "time_first_observed": "str", "violation_county": "str",
            "violation_in_front_of_or_opposite": "str", "house_number": "str", "street_name": "str",
            "intersecting_street": "str", "date_first_observed": "int64", "law_section": "int64",
            "sub_division": "str", "violation_legal_code": "str", "days_parking_in_effect": "str",
            "from_hours_in_effect": "str", "to_hours_in_effect": "str", "vehicle_color": "str",
            "unregistered_vehicle": "str", "vehicle_year": "int64", "meter_number": "str",
            "feet_from_curb": "int64", "violation_post_code": "str", "violation_description": "str",
            "no_standing_or_stopping_violation": "str", "hydrant_violation": "str",
            "double_parking_violation": "str"
        }

        if not os.path.exists(pq_file):
            df = dd.read_csv(self.data_path, dtype=dtype_spec)
            df.to_parquet(pq_file)
        if not os.path.exists(hdf_file):
            df = dd.read_csv(self.data_path, dtype=dtype_spec)
            df.to_hdf(hdf_file, key="data", min_itemsize={"values_block_1": 100})

    def get_file_sizes(
        self,
        pq_file="parking_violations_2024.parquet",
        hdf_file="parking_violations_2024.h5",
    ):
        """
        Runs the task and returns the sizes of the parquet and hdf files.

        Args:
            pq_file (str): The name of the parquet file to save the data to. Defaults to "parking_violations_2024.parquet".
            hdf_file (str): The name of the hdf file to save the data to. Defaults to "parking_violations_2024.h5".

        Returns:
            dict: The sizes of the parquet and hdf files.
        """
        pq_size = os.path.getsize(pq_file)
        hdf_size = os.path.getsize(hdf_file)
        return {"parquet_size": pq_size, "hdf_size": hdf_size}

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

def read_and_process_file(file, reference_cols, base_path='data/', temp_dir='temp/'):
    """
    Read a file, apply column renaming, and save the DataFrame to a temporary location.
    Return the path to the temporary file.
    """
    # Ensure the temporary directory exists
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Read the file with Dask
    df = dd.read_csv(os.path.join(base_path, file), dtype='object')

    # Create mapping and apply renaming
    current_cols = df.columns
    mapping = create_column_mapping(reference_cols, current_cols)
    df = df.rename(columns=mapping)

    # Path for the temporary file
    temp_file = os.path.join(temp_dir, os.path.basename(file))
    
    # Save to Parquet to maintain data types correctly
    df.to_parquet(temp_file)

    return temp_file

def process_all_files(directory='data/', reference_file='2024.csv', temp_dir='temp/'):
    client = Client(
            n_workers=2,
            threads_per_worker=2,
            memory_limit="6.5GB",
            local_directory="/tmp",
        )
    reference_path = os.path.join(directory, reference_file)
    reference_df = pd.read_csv(reference_path, nrows=0)
    reference_cols = list(reference_df.columns)

    files = [f for f in os.listdir(directory) if f.endswith('.csv') and f != reference_file]
    temp_files = [read_and_process_file(f, reference_cols, directory, temp_dir) for f in files]

    # Read all temporary Parquet files into Dask DataFrames and concatenate
    dataframes = [dd.read_parquet(file) for file in temp_files]
    if dataframes:
        combined_df = dd.concat(dataframes, axis=0, ignore_index=True)
        combined_df.to_parquet('data/processed/combined_parking_violations.parquet', write_index=False)
    else:
        print("No dataframes to concatenate.")

    client.close()

def csv_to_hdf5():
    """
    Convert a CSV file to Parquet format.
    """

    client = Client(
            n_workers=2,
            threads_per_worker=2,
            memory_limit="4.5GB",
            local_directory="/tmp",
        )
    print(client.scheduler_info()["services"])
    df = dd.read_csv("data/*.csv", dtype="object")
    df = df.repartition(npartitions=10000)
    df.to_hdf("parking_violations.hdf5", key="data", mode="w", min_itemsize={'Violation Description': 100, 'Plate ID': 50})
    client.close()


if __name__ == "__main__":
    # data_path = "data/parking_violations_2024.csv"
    # task1 = Task1(data_path)
    # print(task1.run())
    #download_data()
    # Create a mapping based on the 2024 file as the standard
    #process_all_files(
    csv_to_hdf5()
    
    #load_all_in_one_parquet()
