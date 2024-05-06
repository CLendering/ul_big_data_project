import dask.dataframe as dd
from dask.distributed import Client
import os

PARKING_VIOLATIONS_2024 = (
    "https://data.cityofnewyork.us/resource/pvqr-7yc4.csv?$limit=13000000"
)


def download_data(data_path="data/parking_violations_2024.csv"):
    """
    Downloads parking violations data from a specified URL and saves it to the given data path.

    Args:
        data_path (str): The path where the downloaded data will be saved. Defaults to "data/parking_violations_2024.csv".

    Returns:
        None
    """
    if not os.path.exists(data_path):
        os.system(f"wget -O {data_path} '{PARKING_VIOLATIONS_2024}'")


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
            "summons_number": "int64",
            "plate_id": "str",
            "registration_state": "str",
            "plate_type": "str",
            "violation_code": "int64",
            "vehicle_body_type": "str",
            "vehicle_make": "str",
            "issuing_agency": "str",
            "street_code1": "int64",
            "street_code2": "int64",
            "street_code3": "int64",
            "vehicle_expiration_date": "int64",
            "violation_location": "str",
            "violation_precinct": "int64",
            "issuer_precinct": "int64",
            "issuer_code": "int64",
            "issuer_command": "str",
            "issuer_squad": "str",
            "violation_time": "str",
            "time_first_observed": "str",
            "violation_county": "str",
            "violation_in_front_of_or_opposite": "str",
            "house_number": "str",
            "street_name": "str",
            "intersecting_street": "str",
            "date_first_observed": "int64",
            "law_section": "int64",
            "sub_division": "str",
            "violation_legal_code": "str",
            "days_parking_in_effect": "str",
            "from_hours_in_effect": "str",
            "to_hours_in_effect": "str",
            "vehicle_color": "str",
            "unregistered_vehicle": "str",
            "vehicle_year": "int64",
            "meter_number": "str",
            "feet_from_curb": "int64",
            "violation_post_code": "str",
            "violation_description": "str",
            "no_standing_or_stopping_violation": "str",
            "hydrant_violation": "str",
            "double_parking_violation": "str",
        }

        if not os.path.exists(pq_file):
            df = dd.read_csv(self.data_path, dtype=dtype_spec)
            df.to_parquet(pq_file)
        if not os.path.exists(hdf_file):
            df = dd.read_csv(self.data_path, dtype=dtype_spec)
            df.to_hdf(hdf_file, key="data", min_itemsize={"values_block_1": 100})

    def run(
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


if __name__ == "__main__":
    data_path = "data/parking_violations_2024.csv"
    task1 = Task1(data_path)
    print(task1.run())
