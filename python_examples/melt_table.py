import datetime
import pandas as pd
import os
import argparse
from typing import Optional


def read_csv(input_csv: str) -> pd.DataFrame:
    """Reads a CSV file into a DataFrame."""
    try:
        df: pd.DataFrame = pd.read_csv(input_csv)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {input_csv} was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")
    except pd.errors.ParserError:
        raise ValueError("The file could not be parsed.")
    return df


def preprocess_dataframe(df: pd.DataFrame, trim: int, time_format: str) -> pd.DataFrame:
    """Preprocesses the DataFrame by converting column names to lowercase and
    adjusting the time column to unix time in nanoseconds."""
    df.columns = df.columns.str.lower()
    if "time" not in df.columns:
        raise ValueError("The required column 'time' is missing from the input file.")

    if time_format == "datetime":
        try:
            df["unix_time_ns"] = pd.to_datetime(df["time"]).astype("int64") + (trim * 10**9)
        except ValueError as e:
            raise ValueError(f"Error converting 'time' column: {e}")
    elif time_format == "unix_ns":
        try:
            df["unix_time_ns"] = df["time"].astype("int64") + (trim * 10**9)
        except ValueError as e:
            raise ValueError(f"Error converting 'time' column: {e}")
    else:
        raise ValueError("Invalid time_format. Use 'datetime' or 'unix_ns'.")

    df.drop("time", axis=1, inplace=True)
    return df


def drop_unwanted_columns(df: pd.DataFrame, drop_columns: str) -> pd.DataFrame:
    """Drops columns from the DataFrame that match the specified pattern."""
    if drop_columns:
        df = df.loc[:, ~df.columns.str.contains(drop_columns)]
    return df


def melt_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Melts DataFrame to long format."""
    value_vars: pd.Index = df.drop(columns=["unix_time_ns"]).columns
    df_melt: pd.DataFrame = pd.melt(df, id_vars="unix_time_ns", value_vars=value_vars)
    df_melt.columns = ["unix_time_ns", "param_key", "param_value"]
    return df_melt


def filter_dataframe(
    df_melt: pd.DataFrame, filter_start: Optional[int], filter_end: Optional[int]
) -> pd.DataFrame:
    """Filters the melted DataFrame based on the specified nanosecond range."""
    if filter_start is not None:
        df_melt = df_melt[df_melt["unix_time_ns"] >= str(filter_start)]
    if filter_end is not None:
        df_melt = df_melt[df_melt["unix_time_ns"] <= str(filter_end)]
    return df_melt


def save_dataframe(df_melt: pd.DataFrame, output_folder: str) -> str:
    """Saves the DataFrame to a CSV file."""
    current_datetime: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    output_file_name: str = os.path.join(output_folder, f"output_melt_table_{current_datetime}.csv")
    df_melt.to_csv(output_file_name, index=False, header=True)
    return output_file_name


def process_data(
    input_csv: str,
    ns_trim: int = 0,
    drop_columns: str = "",
    filter_start: Optional[int] = None,
    filter_end: Optional[int] = None,
    output_folder: str = "",
    time_format: str = "datetime",  # Default to datetime
) -> None:
    """Main function to process the IADS data."""
    try:
        df: pd.DataFrame = read_csv(input_csv)
        df = preprocess_dataframe(df, ns_trim, time_format)
        df = drop_unwanted_columns(df, drop_columns)
        df_melt: pd.DataFrame = melt_dataframe(df)
        df_melt = filter_dataframe(df_melt, filter_start, filter_end)
        df_melt = df_melt.sort_values(by="unix_time_ns").reset_index(drop=True)
        output_file_name: str = save_dataframe(df_melt, output_folder)
        print(f"Output saved to {output_file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Melt columnar data to long format.")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file")
    parser.add_argument(
        "--trim",
        type=int,
        default=0,
        help=(
            "Second trim value (positive or negative integer) e.g. "
            "1703980800 for IADS data 1970/1/1 to 2024/1/0"
        ),
    )
    parser.add_argument(
        "--drop_columns",
        type=str,
        default="",
        help='String to identify columns to drop (e.g., "seq_num|timestamp")',
    )
    parser.add_argument("--filter_start", type=int, help="Start of the filter nanosecond range")
    parser.add_argument("--filter_end", type=int, help="End of the filter nanosecond range")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="",
        help="Output folder path (default is current working directory)",
    )
    parser.add_argument(
        "--time_format",
        type=str,
        choices=["datetime", "unix_ns"],
        default="datetime",
        help='Format of the time column: "datetime" for datetime strings or "unix_ns" '
        'for UNIX timestamps in nanoseconds.',
    )

    args = parser.parse_args()

    try:
        process_data(
            args.input_csv,
            args.trim,
            args.drop_columns,
            args.filter_start,
            args.filter_end,
            args.output_folder,
            args.time_format,
        )
    except Exception as e:
        print(f"An error occurred: {e}")
