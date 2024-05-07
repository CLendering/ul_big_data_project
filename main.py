import dask.dataframe as dd
from dask.distributed import Client
import os
import urllib

PARKING_VIOLATIONS_2024 = (
    "https://data.cityofnewyork.us/resource/pvqr-7yc4.csv?$limit=13000000"
)
PARKING_VIOLATIONS_2023 = (
    "https://data.cityofnewyork.us/api/views/869v-vr48/rows.csv?accessType=DOWNLOAD"
)
PARKING_VIOLATIONS_2022 = (
    "https://data.cityofnewyork.us/api/views/7mxj-7a6y/rows.csv?accessType=DOWNLOAD"
)
PARKING_VIOLATIONS_2020 = (
    "https://data.cityofnewyork.us/api/views/p7t3-5i9s/rows.csv?accessType=DOWNLOAD"
)
PARKING_VIOLATIONS_2019 = (
    "https://data.cityofnewyork.us/api/views/faiq-9dfq/rows.csv?accessType=DOWNLOAD"
)
PARKING_VIOLATIONS_2018 = (
    "https://data.cityofnewyork.us/api/views/a5td-mswe/rows.csv?accessType=DOWNLOAD"
)
PARKING_VIOLATIONS_2017 = (
    "https://data.cityofnewyork.us/api/views/2bnn-yakx/rows.csv?accessType=DOWNLOAD"
)
PARKING_VIOLATIONS_2016 = (
    "https://data.cityofnewyork.us/api/views/kiv2-tbus/rows.csv?accessType=DOWNLOAD"
)
PARKING_VIOLATIONS_2015 = (
    "https://data.cityofnewyork.us/api/views/c284-tqph/rows.csv?accessType=DOWNLOAD"
)
PARKING_VIOLATIONS_2014 = (
    "https://data.cityofnewyork.us/api/views/jt7v-77mi/rows.csv?accessType=DOWNLOAD"
)

parking_violations = [
    PARKING_VIOLATIONS_2024,
    PARKING_VIOLATIONS_2023,
    PARKING_VIOLATIONS_2022,
    PARKING_VIOLATIONS_2020,
    PARKING_VIOLATIONS_2019,
    PARKING_VIOLATIONS_2018,
    PARKING_VIOLATIONS_2017,
    PARKING_VIOLATIONS_2016,
    PARKING_VIOLATIONS_2015,
    PARKING_VIOLATIONS_2014,
]


def download_data(data_folder="data"):
    """
    Downloads parking violations data from a specified URL and saves it to the given data path.

    Args:
        data_folder (str): The folder where the downloaded data will be saved. Defaults to "data".

    Returns:
        None
    """
    year = 2024
    for url in parking_violations:
        file_name = f"parking_violations_{year}.csv"
        file_path = os.path.join(data_folder, file_name)
        if not os.path.exists(file_path):
            urllib.request.urlretrieve(url, file_path)
        year -= 1


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
    # data_path = "data/parking_violations_2024.csv"
    # task1 = Task1(data_path)
    # print(task1.run())
    download_data()
