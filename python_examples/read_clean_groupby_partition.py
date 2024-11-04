import pandas as pd
import fastparquet as fp
from typing import Optional


class LapTimeProcessor:
    """
    A class to process lap time data and driver details, merge them,
    and save the results to a partitioned Parquet file.

    Attributes:
        df_bahrain (pd.DataFrame): DataFrame containing Bahrain lap time data.
        df_drivers (pd.DataFrame): DataFrame containing driver details.
        df_merge (pd.DataFrame): DataFrame resulting from merging df_bahrain and df_drivers.
    """

    def __init__(self):
        self.df_bahrain: Optional[pd.DataFrame] = None
        self.df_drivers: Optional[pd.DataFrame] = None
        self.df_merge: Optional[pd.DataFrame] = None

    def load_data(self, lap_times_csv: str, drivers_json: str) -> None:
        """
        Load lap time data from a CSV file and driver details from a JSON file.

        Args:
            lap_times_csv (str): Path to the CSV file containing Bahrain lap times.
            drivers_json (str): Path to the JSON file containing driver details.
        """
        self.df_bahrain = pd.read_csv(lap_times_csv)
        self.df_drivers = pd.read_json(drivers_json, lines=True)
        self.df_bahrain = self.df_bahrain.dropna(subset=["Laptimes"])

    def merge_data(self) -> None:
        """
        Merge the Bahrain lap times DataFrame with the driver details DataFrame.
        """
        self.df_merge = pd.merge(
            self.df_bahrain,
            self.df_drivers,
            left_on="DriverNumber",
            right_on="driver_name",
            how="left",
        )
        self.df_merge["Laptimes"] = pd.to_timedelta(self.df_merge["Laptimes"])

    def sort_data(self) -> pd.DataFrame:
        """
        Sort the merged DataFrame by lap times.

        Returns:
            pd.DataFrame: The sorted DataFrame.
        """
        sorted_df = self.df_merge.sort_values(by="Laptimes")
        return sorted_df

    def find_fastest_laps(self) -> pd.DataFrame:
        """
        Find the fastest lap times for each driver.

        Returns:
            pd.DataFrame: DataFrame with the fastest lap times per driver.
        """
        fastest_laps = self.df_merge.groupby("DriverNumber", as_index=False).agg(
            {"Laptimes": "min"}
        )
        return fastest_laps

    def save_to_parquet(self, output_dir: str) -> None:
        """
        Save the merged DataFrame to a partitioned Parquet file.

        Args:
            output_dir (str): Directory to save the partitioned Parquet files.
        """
        fp.write(output_dir, self.df_merge, partition_on=["DriverNumber", "Compound"])

    def read_parquet(self, input_dir: str) -> pd.DataFrame:
        """
        Read a partitioned Parquet file into a DataFrame.

        Args:
            input_dir (str): Directory containing the partitioned Parquet files.

        Returns:
            pd.DataFrame: DataFrame read from the partitioned Parquet files.
        """
        return fp.ParquetFile(input_dir).to_pandas()


# Example usage
if __name__ == "__main__":
    processor = LapTimeProcessor()
    processor.load_data("bahrain_laptimes.csv", "driver_details.json")
    processor.merge_data()

    print("Merged DataFrame:")
    print(processor.df_merge)

    sorted_df = processor.sort_data()
    print("\nSorted DataFrame by Laptimes:")
    print(sorted_df)

    fastest_laps = processor.find_fastest_laps()
    print("\nFastest lap times per driver:")
    print(fastest_laps)

    processor.save_to_parquet("partitioned_laptimes.parquet")

    df_read_back = processor.read_parquet("partitioned_laptimes.parquet")
    print("\nDataFrame read back from partitioned Parquet file:")
    print(df_read_back)
