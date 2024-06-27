import dask.dataframe as dd
from dask.distributed import Client
import pandas as pd
import os
from rapidfuzz import process
from collections import defaultdict
from pyarrow import parquet as pq
from tables import NaturalNameWarning
import warnings
import json
import traceback
import geopandas as gpd
from geopy.geocoders import Nominatim
import time

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
    client = Client(
        n_workers=2,
        threads_per_worker=2,
        memory_limit="5.5GB",
        local_directory="/tmp",
    )
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
    default_min_itemsize = {
        "issuing_agency": 10,
        "sub_division": 10,
        "house_number": 20,
        "violation_location": 10,
        "violation_county": 10,
        "violation_description": 80,
        "vehicle_expiration_date": 30,
        "date_first_observed": 30,
        "issue_date": 30,
        "issuer_command": 10,
        "feet_from_curb": 10,
        "street_name": 300,
        "summons_number": 10,
        "plate_id": 20,
        "registration_state": 2,
        "plate_type": 3,
        "violation_code": 2,
        "vehicle_body_type": 4,
        "vehicle_make": 5,
        "street_code1": 5,
        "street_code2": 5,
        "street_code3": 5,
        "violation_precinct": 3,
        "issuer_precinct": 3,
        "issuer_code": 6,
        "issuer_squad": 4,
        "violation_time": 5,
        "time_first_observed": 5,
        "violation_in_front_of_or_opposite": 1,
        "intersecting_street": 20,
        "law_section": 4,
        "violation_legal_code": 1,
        "days_parking_in_effect": 7,
        "from_hours_in_effect": 5,
        "to_hours_in_effect": 5,
        "vehicle_color": 5,
        "unregistered_vehicle": 1,
        "vehicle_year": 4,
        "meter_number": 8,
        "violation_post_code": 300,
        "no_standing_or_stopping_violation": 40,
        "hydrant_violation": 25,
        "double_parking_violation": 36,
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

                # Limit string columns to 300 characters
                df = df.map(lambda x: x[:300] if isinstance(x, str) else x)

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
    parquet_size = sum(
        os.path.getsize(os.path.join(parquet_directory, f)) for f in parquet_files
    )

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
    compare_file_sizes(
        "data/processed/combined_parking_violations.h5",
        "data/processed/combined_parking_violations.parquet",
    )


def convert_dbf_to_csv(dbf_path, csv_path):
    """
    Converts a .dbf file to a .csv file.

    Parameters:
    dbf_path (str): The path to the .dbf file.
    csv_path (str): The path where the .csv file will be saved.
    """
    # Reading the .dbf file using geopandas
    dbf_data = gpd.read_file(dbf_path)

    # Saving the dataframe to a CSV file
    dbf_data.to_csv(csv_path, index=False)
    print(f"Data has been successfully converted to {csv_path}")


# NOT WORKING: API rate limit exceeded
def geo_py_zip_code_lookup(address):
    """
    Looks up the zip code for a given address using GeoPy.

    Parameters:
    address (str): The address for which the zip code will be looked up.

    Returns:
    str: The zip code for the address.
    """
    geolocator = Nominatim(user_agent="abcd", timeout=10)
    from geopy.extra.rate_limiter import RateLimiter

    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geocode(address + ", New York, USA", addressdetails=True)
    if location:
        print(location.address.split(",")[-2].strip())
        return location.address.split(",")[-2].strip()
    else:
        return None


# NOT WORKING: API rate limit exceeded
def create_zip_code_dataset():

    client = Client(
        n_workers=2, threads_per_worker=2, memory_limit="5.5GB", local_directory="/tmp"
    )
    df = dd.read_parquet(
        "data/processed/combined_parking_violations.parquet",
        columns=["street_name", "house_number"],
    )
    df = df.dropna()

    # Combine the address and house number columns
    df["full_address"] = df["street_name"] + " " + df["house_number"]

    # get unique addresses
    unique_addresses = df["full_address"].unique().compute()

    print(f"Found {len(unique_addresses)} unique addresses")

    # get the zip code for each address
    zip_codes = [geo_py_zip_code_lookup(address) for address in unique_addresses]

    # Create a DataFrame with the addresses and zip codes
    zip_code_df = pd.DataFrame(
        {"full_address": unique_addresses, "zip_code": zip_codes}
    )

    # Save the DataFrame to a CSV file
    zip_code_df.to_csv("data/processed/zip_codes.csv", index=False)
    print("Zip codes have been successfully saved to 'data/processed/zip_codes.csv'")

    client.close()


def get_similar_address_table(
    output_file="similar_address_table.csv", batch_size=10000
):
    client = Client(
        n_workers=2, threads_per_worker=2, memory_limit="5.5GB", local_directory="/tmp"
    )
    # Load the address reference dataset
    address_reference = dd.read_csv(
        "Address_Point.csv", usecols=["FULL_STREE", "ZIPCODE"], dtype="object"
    ).persist()

    address_reference = address_reference.dropna().compute()
    unique_address_reference = address_reference["FULL_STREE"].unique()

    # Load the address dataset
    address_dataset = dd.read_parquet(
        "data/processed/combined_parking_violations.parquet",
        columns=["street_name", "house_number"],
    )

    address_dataset = address_dataset.dropna()

    # Create the full address field
    address_dataset["full_address"] = (
        address_dataset["street_name"] + " " + address_dataset["house_number"]
    )
    unique_full_address = address_dataset["full_address"].unique().persist()

    # Initialize a list to collect rows
    rows = []

    # Open the output file and write the header
    with open(output_file, "w") as f:
        f.write("Address,Similar_Address, zip_code\n")

    # Find the similar addresses and write in batches]

    start_time = time.time()
    for i, address in enumerate(unique_full_address):
        similar_address = process.extractOne(address, unique_address_reference)[0]
        zip_code = address_reference[
            address_reference["FULL_STREE"] == similar_address
        ]["ZIPCODE"].values[0]
        rows.append(
            {
                "Address": address,
                "Similar_Address": similar_address,
                "zip_code": zip_code,
            }
        )

        # Write to disk in batches
        if (i + 1) % batch_size == 0:
            batch_df = pd.DataFrame(rows)
            batch_df.to_csv(output_file, mode="a", header=False, index=False)
            rows = []  # Reset rows list to free up memory
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time: {elapsed_time} seconds", flush=True)
            start_time = end_time

    # Write any remaining rows to disk
    if rows:
        batch_df = pd.DataFrame(rows)
        batch_df.to_csv(output_file, mode="a", header=False, index=False)

    print(f"Results written to {output_file}")
    client.close()


def get_school_counts():
    # Load the school data
    school_data = dd.read_csv(
        "Public_School_Locations/Public_Schools_Points_2011-2012A.csv", dtype="object"
    )

    # Count the number of schools in each borough
    school_counts = school_data["BORO"].value_counts().compute()

    return school_counts


def get_parking_violations_counties():
    # Load the parking violations data
    parking_data = dd.read_parquet("data/processed/combined_parking_violations.parquet")

    # Get the unique counties in the data
    counties = parking_data["violation_county"].value_counts().compute()

    return counties


def standardize_borough_names():
    # Load the parking violations data
    parking_data = dd.read_parquet("data/processed/combined_parking_violations.parquet")

    borough_mapping = {
        "Bronx": "X",
        "BRONX": "X",
        "BX": "X",
        "K": "K",
        "K F": "K",
        "Kings": "K",
        "KINGS": "K",
        "BK": "K",
        "MAN": "M",
        "MH": "M",
        "MN": "M",
        "NEW Y": "M",
        "NEWY": "M",
        "NY": "M",
        "NYC": "M",
        "QU": "Q",
        "Qns": "Q",
        "QN": "Q",
        "QNS": "Q",
        "QUEEN": "Q",
        "R": "R",
        "Rich": "R",
        "RICH": "R",
        "RICHM": "R",
        "RC": "R",
    }

    parking_data["violation_county"] = parking_data["violation_county"].map(
        borough_mapping, meta=("violation_county", "object")
    )

    parking_data.to_parquet(
        "data/processed/combined_parking_violations.parquet", write_index=False
    )


def join_parking_schools_on_borough():

    client = Client(
        n_workers=12, threads_per_worker=1, memory_limit="7", local_directory="/tmp"
    )
    # Print the client address
    print(client)
    # Load the parking violations data
    parking_data = dd.read_parquet("data/processed/combined_parking_violations.parquet")

    # Load the school data
    school_data = dd.read_csv(
        "Public_School_Locations/Public_Schools_Points_2011-2012A.csv", dtype="object"
    )

    parking_data = parking_data.repartition(npartitions=10000)
    school_data = school_data.repartition(npartitions=10)

    # Make the join
    joined_data = dd.merge(
        parking_data, school_data, left_on="violation_county", right_on="BORO"
    )

    joined_data.to_parquet(
        "data/processed/parking_schools_joined.parquet", write_index=False
    )

    client.close()


def clean_address_data():
    client = Client(n_workers=2, threads_per_worker=2, memory_limit="5.5GB")

    parking_data = dd.read_parquet("data/processed/combined_parking_violations.parquet")

    # Drop rows with missing values
    parking_data = parking_data.dropna(subset=["house_number", "street_name"])

    # Make all characters uppercase
    parking_data["house_number"] = parking_data["house_number"].str.upper()
    parking_data["street_name"] = parking_data["street_name"].str.upper()

    # Remove odd characters
    parking_data["house_number"] = parking_data["house_number"].str.replace(
        "[^A-Z0-9\s]", "", regex=True
    )
    parking_data["street_name"] = parking_data["street_name"].str.replace(
        "[^A-Z0-9\s]", "", regex=True
    )

    # Remove leading and trailing whitespace
    parking_data["house_number"] = parking_data["house_number"].str.strip()
    parking_data["street_name"] = parking_data["street_name"].str.strip()

    # Check for empty strings
    parking_data = parking_data[
        (parking_data["house_number"] != "") & (parking_data["street_name"] != "")
    ]

    parking_data.to_parquet(
        "data/processed/cleaned_parking_violations.parquet", write_index=False
    )

    client.close()

# DEPRECATED: Join does not yeild sufficient results
def get_postcodes():
    # Initialize a Dask client with optimal configuration
    client = Client(n_workers=8, threads_per_worker=2, memory_limit='16GB')

    # Read data
    parking_data = dd.read_parquet("data/processed/cleaned_parking_violations.parquet")
    address_reference = dd.read_csv("Address_Point.csv", usecols=["FULL_STREE", "ZIPCODE"], dtype="object").persist()

    # Combine the address columns and convert to categorical
    address_reference["full_address"] = address_reference["FULL_STREE"].astype('category').cat.as_ordered()
    parking_data["full_address"] = (parking_data["street_name"] + " " + parking_data["house_number"]).astype('category').cat.as_ordered()

    # Repartition the data
    parking_data = parking_data.repartition(npartitions=500)
    address_reference = address_reference.repartition(npartitions=50)

    # Set the index to optimize the join
    parking_data = parking_data.set_index("full_address", sorted=True, drop=False)
    address_reference = address_reference.set_index("full_address", sorted=True, drop=False)

    # Persist intermediate results to memory to avoid recomputation
    parking_data = parking_data.persist()
    address_reference = address_reference.persist()

    # Perform the left join
    joined_data = dd.merge(parking_data, address_reference, on="full_address", how="left")

    # Save the joined data
    joined_data.to_parquet("data/processed/parking_violations_with_postcodes.parquet", write_index=False)

    # Close the client
    client.close()

# DEPRECATED: Join does not yeild sufficient results
def inspect_joined_data():
    # Load the joined data
    joined_data = dd.read_parquet("/home/camile/big_data_project/data/processed/parking_violations_with_postcodes.parquet").dropna()

    # Display the first few rows
    print(joined_data.head())

    # Display the number of rows
    print(f"Number of rows: {len(joined_data)}")

    # Display the number of unique zip codes
    print(f"Number of unique zip codes: {len(joined_data['ZIPCODE'].unique())}")

def standardize_zip_codes():
    data_path = "data/processed/cleaned_parking_violations.parquet/*.parquet"
    
    # Initialize the Dask client
    client = Client(n_workers=2, threads_per_worker=2, memory_limit="5.5GB")
    
    # Load the data
    parking_data = dd.read_parquet(data_path, engine="pyarrow")
    
    # Drop rows with missing values in 'violation_post_code'
    parking_data = parking_data.dropna(subset=["violation_post_code"])
    print("After dropping NA:", parking_data.shape[0].compute())

    # Remove leading and trailing whitespace
    parking_data["violation_post_code"] = parking_data["violation_post_code"].str.strip()
    print("After stripping whitespace:", parking_data.shape[0].compute())

    # Remove rows that are not numeric
    parking_data = parking_data[parking_data["violation_post_code"].str.isnumeric()]
    print("After removing non-numeric:", parking_data.shape[0].compute())
    
    # Define the function to handle the zip code transformation
    def transform_zip_code(x):
        if len(x) == 2:
            return "100" + x
        else:
            return x
    
    # Apply the transformation using map_partitions 
    parking_data["violation_post_code"] = parking_data["violation_post_code"].map_partitions(
        lambda df: df.apply(transform_zip_code),
        meta=('violation_post_code', 'object')
    )
    print("After transforming zip codes:", parking_data.shape[0].compute())

    # Ensure that the zip codes are 5 characters
    parking_data = parking_data[parking_data["violation_post_code"].str.len() == 5]
    print("After filtering zip codes with length 5:", parking_data.shape[0].compute())
    
    # Save the cleaned data
    parking_data.to_parquet("data/processed/cleaned_zip_parking_violations.parquet", write_index=False)
    
    # Close the Dask client
    client.close()

def inspect_cleaned_zip_data():
    # Load the cleaned data
    cleaned_data = dd.read_parquet("data/processed/cleaned_zip_parking_violations.parquet")
    
    # Display the first few rows post codes and street names
    print(cleaned_data[["violation_post_code", "street_name", "violation_county"]].head(10, npartitions=100))  
    # Display the number of rows
    print(f"Number of rows: {len(cleaned_data)}")
    
    # Display the number of unique zip codes
    print(f"Number of unique zip codes: {len(cleaned_data['violation_post_code'].unique())}")




if __name__ == "__main__":
    # standardize_zip_codes()
    inspect_cleaned_zip_data()
