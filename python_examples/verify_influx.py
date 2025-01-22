import argparse
import influxdb_client_3 as InfluxDBClient3
from influxdb_client_3 import flight_client_options
import certifi
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
import pickle
from pandas.io.formats import excel
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from itertools import islice
import time
import math

exception_counter = 0


def read_certificate():
    """Read the certificate file."""
    with open(certifi.where(), "r") as fh:
        return fh.read()


def read_influxdb_credentials(credentials_path):
    """Read InfluxDB credentials from a file."""
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"Credentials file '{
                                credentials_path}' not found")

    with open(credentials_path, "r") as f:
        lines = f.read().splitlines()
        return {
            "host": lines[0].strip(),
            "token": lines[1].strip(),
            "database": lines[2].strip(),
        }


def create_influxdb_client(token, host, database, cert):
    """Create and return an InfluxDB client."""
    return InfluxDBClient3.InfluxDBClient3(
        token=token,
        host=host,
        database=database,
        flight_client_options=flight_client_options(tls_root_certs=cert),
    )


def load_pickle(pickle_path):
    """Load data and metadata from a pickle file."""
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            pickle_data = pickle.load(f)

        tables_w_data_dict = pickle_data.get("tables_w_data_dict", {})
        tables_w_data_list = pickle_data.get("tables_w_data_list", [])
        tables_wo_data_list = pickle_data.get("tables_wo_data_list", [])
        start_time = pickle_data.get("start_time", "N/A")
        end_time = pickle_data.get("end_time", "N/A")
        config = pickle_data.get("config", "N/A")
        database = pickle_data.get("database", "N/A")

        if tables_w_data_dict:
            print("\nInflux tables loaded from pickle file:")
            return (
                tables_w_data_dict,
                tables_w_data_list,
                tables_wo_data_list,
                start_time,
                end_time,
                config,
                database,
                True,
            )
        else:
            print("\nWARNING: Pickle file contains no data")
            return ({}, [], [], "N/A", "N/A", "N/A", "N/A", False)
    else:
        print("\nWARNING: Pickle file not found")
        return ({}, [], [], "N/A", "N/A", "N/A", "N/A", False)


def query_influx_measurements(client):
    """Query and return InfluxDB measurements."""
    query = "SHOW MEASUREMENTS"
    result = client.query(query=query, language="influxql")
    result_df = result.to_pandas()
    found_tables_list = result_df["name"].to_list()

    return found_tables_list


def query_table(client, table, start_time, end_time):
    """Query a single InfluxDB table."""
    try:
        query = f"SELECT * FROM '{table}' WHERE time >= '{
            start_time}' AND time <= '{end_time}'"
        table_data = client.query(query=query, language="sql").to_pandas()
        return table, table_data
    except Exception as e:
        print(f"Critical Error: Error querying table {table}: {e}")
        sys.exit(1)


def query_table_by_row(client, table, start_time, end_time, attempt):
    """Query a table in batches by time and LIMIT, adjusting batch size on retries."""
    initial_batch_size = 100000
    batch_size = initial_batch_size // (2**attempt)
    start_time_cursor = start_time
    all_data_dfs = []

    while True:
        query = (
            f"SELECT * FROM \"{table}\" WHERE time >= '{start_time_cursor}' "
            f"AND time <= '{end_time}' ORDER BY time LIMIT {batch_size}"
        )

        try:
            print(f"Executing query: {query} with batch size {batch_size}")
            table_data = client.query(query=query, language="sql")
            table_data_df = table_data.to_pandas()

            if table_data_df.empty:
                break

            all_data_dfs.append(table_data_df)
            start_time_cursor = table_data_df["time"].iloc[-1] + pd.Timedelta(
                nanoseconds=1
            )

            if len(table_data_df) < batch_size:
                break

        except Exception as e:
            print(
                f"WARNING: Error querying table '{
                    table}' with start time {start_time_cursor}: {e}"
            )
            raise

    if all_data_dfs:
        combined_df = pd.concat(all_data_dfs, ignore_index=True)
    else:
        combined_df = pd.DataFrame()

    return table, combined_df


def query_table_with_retries(client, table, start_time, end_time):
    """Query a single InfluxDB table with retry on failure."""
    max_retries = 3
    retry_delay = 5

    for attempt in range(max_retries):
        try:
            return query_table_by_row(client, table, start_time, end_time, attempt)
        except Exception as e:
            print(f"Error querying table {table}: {e}")
            if attempt < max_retries - 1:
                print(
                    f"Retrying table {
                        table}: (Attempt {attempt + 2}/{max_retries}) "
                    f"after {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                print(f"Critical Error: Max retries reached for table {
                      table}: {e}")
                raise


def chunked(iterable, n):
    """Yield successive n-sized chunks from an iterable."""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def query_influx_timestamps_normal(client, found_tables_list, start_time, end_time):
    """Query InfluxDB for timestamps and categorise tables based on data presence."""
    print("Method: 'normal'")
    tables_w_data_dict = {}
    tables_w_data_list = []
    tables_wo_data_list = []

    total_tables = len(found_tables_list)
    for index, table in enumerate(found_tables_list):
        print(f"Querying {index + 1}/{total_tables}: '{table}'")
        query = (f"SELECT * FROM \"{table}\" WHERE time >= '{start_time}'"
                 f"AND time <= '{end_time}'")
        try:
            table_data = client.query(query=query, language="sql")
            table_data_df = table_data.to_pandas()

            if not table_data_df.empty:
                tables_w_data_dict[table] = table_data_df
                tables_w_data_list.append(table)
            else:
                tables_wo_data_list.append(table)
        except Exception as e:
            print(f"Critical Error: Error querying table {table}: {e}")
            sys.exit(1)

    print(f"Number of tables with data: {len(tables_w_data_dict)}")
    for table, df in tables_w_data_dict.items():
        print(f"Table: {table}, DataFrame Shape: {df.shape}")

    return tables_w_data_dict, tables_w_data_list, tables_wo_data_list


def query_influx_timestamps_tpemaprc(client, found_tables_list, start_time, end_time):
    """Query InfluxDB for timestamps and categorise tables based on data presence."""
    print("Method: ThreadPoolExecutor.map()")

    tables_w_data_dict = {}
    tables_w_data_list = []
    tables_wo_data_list = []

    num_cores = os.cpu_count()
    num_workers = max(num_cores - 1, 1)
    total_tables = len(found_tables_list)

    try:
        completed = 0
        chunk_size = 100
        chunks = list(chunked(found_tables_list, chunk_size))
        num_chunks = len(chunks)

        for chunk_index, chunk in enumerate(chunks, start=1):
            print(
                f"Processing chunk {
                    chunk_index}/{num_chunks} with {len(chunk)} tables."
            )
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = executor.map(
                    lambda table: query_table_with_retries(
                        client, table, start_time, end_time
                    ),
                    chunk,
                )

                for table, table_data_df in results:
                    if not table_data_df.empty:
                        tables_w_data_dict[table] = table_data_df
                        tables_w_data_list.append(table)
                    else:
                        tables_wo_data_list.append(table)

                    completed += 1
                    print(
                        f"Finished querying {
                            completed}/{total_tables}: '{table}' "
                        f"(Chunk {
                            chunk_index}/{num_chunks} of chunk size: {chunk_size})"
                    )

        return tables_w_data_dict, tables_w_data_list, tables_wo_data_list

    except Exception as e:
        print(f"Critical error: {e}")
        raise


def query_influx_timestamps_tpefutr(client, found_tables_list, start_time, end_time):
    """Query InfluxDB for timestamps and categorize tables based on data presence."""
    print("Method: ThreadPoolExecuter.submit()")
    tables_w_data_dict = {}
    tables_w_data_list = []
    tables_wo_data_list = []

    num_cores = os.cpu_count()
    num_workers = max(num_cores - 1, 1)
    total_tables = len(found_tables_list)

    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    query_table_with_retries, client, table, start_time, end_time
                ): table
                for table in found_tables_list
            }

            completed = 0
            for future in as_completed(futures):
                table = futures[future]
                try:
                    table, table_data_df = future.result()

                    if not table_data_df.empty:
                        tables_w_data_dict[table] = table_data_df
                        tables_w_data_list.append(table)
                    else:
                        tables_wo_data_list.append(table)

                    completed += 1
                    print(f"Finished querying {
                          completed}/{total_tables}: '{table}'")

                except Exception as e:
                    print(f"Error processing table {table}: {e}")
                    raise

        ordered_tables_w_data_dict = {
            table: tables_w_data_dict[table]
            for table in found_tables_list
            if table in tables_w_data_dict
        }

        return ordered_tables_w_data_dict, tables_w_data_list, tables_wo_data_list

    except Exception as e:
        print(f"Critical error: {e}")
        raise


def save_to_pickle(
    output_path,
    tables_w_data_dict,
    tables_w_data_list,
    tables_wo_data_list,
    start_time,
    end_time,
    config,
    database,
):
    """Save data and metadata to a pickle file."""
    with open(output_path, "wb") as f:
        pickle.dump(
            {
                "tables_w_data_dict": tables_w_data_dict,
                "tables_w_data_list": tables_w_data_list,
                "tables_wo_data_list": tables_wo_data_list,
                "start_time": start_time,
                "end_time": end_time,
                "config": config,
                "database": database,
            },
            f,
        )
    print(f"\nSaved Influx data and metadata to pickle file '{output_path}'")


def calculate_intervals(lru_name, lru_df):
    """Calculate intervals between timestamps in a DataFrame."""
    lru_df["time"] = pd.to_datetime(lru_df["time"], errors="coerce")
    lru_df = lru_df.sort_values(by="time")
    intervals_np = lru_df["time"].diff().dt.total_seconds().to_numpy()[
        1:] * 1000
    invalid_intervals = intervals_np[np.isnan(intervals_np)]

    if len(invalid_intervals) > 0:
        print(f"LRU {lru_name} contains {
              len(invalid_intervals)} invalid intervals")

    return intervals_np


def check_for_gaps(tables_w_data_dict, icd_message_data_df):
    """Analyse message intervals written in Influx tables."""
    print("\nChecking intervals within Influx tables")

    interval_analysis = []
    selected_intervals = {}

    for table, df in tables_w_data_dict.items():
        unique_lru_names = df["hub_lru_name"].unique()

        for lru_name in unique_lru_names:
            lru_df = df[df["hub_lru_name"] == lru_name].copy()
            intervals_np = calculate_intervals(lru_name, lru_df)

            if len(intervals_np) > 0:
                mean_interval = np.round(np.mean(intervals_np), 2)
                max_interval = np.round(np.max(intervals_np), 2)
                min_interval = np.round(np.min(intervals_np), 2)
                std_interval = np.round(np.std(intervals_np), 1)
                coefficient_of_variation = np.round(
                    (std_interval / mean_interval) * 100, 1
                )
                if std_interval > 0:
                    max_z_score = np.round(
                        (max_interval - mean_interval) / std_interval, 1
                    )
                    min_z_score = np.round(
                        (min_interval - mean_interval) / std_interval, 1
                    )
                else:
                    max_z_score = np.nan
                    min_z_score = np.nan

                late_intervals_ct = np.sum(intervals_np > 1.5 * mean_interval)
                late_intervals_ct_pc = np.round(
                    (late_intervals_ct / len(intervals_np)) * 100, 3
                )
                late_intervals_ratio = f"{
                    late_intervals_ct}/{len(intervals_np)}"

                icd_rows = icd_message_data_df[
                    icd_message_data_df["Message Name"] == table
                ]
                if not icd_rows.empty:
                    icd_interval = icd_rows["Transmit Interval (ms)"].values[0]
                    max_occurrences = icd_rows["Max Occurrences"].values[0]

                    icd_interval = pd.to_numeric(icd_interval, errors="coerce")
                    max_occurrences = pd.to_numeric(
                        max_occurrences, errors="coerce")

                    if pd.isna(icd_interval):
                        icd_interval = "ICD found but no transmit interval available"
                        compare_mean_percent = "N/A"
                        compare_max_percent = "N/A"
                        compare_min_percent = "N/A"
                        interval_check = "N/A"
                        icd_interval_refactored = "N/A"
                    elif pd.isna(max_occurrences):
                        compare_mean_percent = "N/A"
                        compare_max_percent = "N/A"
                        compare_min_percent = "N/A"
                        interval_check = "N/A"
                        icd_interval_refactored = (
                            "Transmit interval found but no occurences"
                        )
                    else:
                        icd_interval_refactored = np.round(
                            icd_interval / max_occurrences, 2
                        )
                        compare_mean_percent = (
                            mean_interval / icd_interval_refactored
                        ) * 100
                        compare_max_percent = (
                            max_interval / icd_interval_refactored
                        ) * 100
                        compare_min_percent = (
                            min_interval / icd_interval_refactored
                        ) * 100

                        if (
                            compare_mean_percent < 90
                            or compare_mean_percent > 110
                            or compare_max_percent < 90
                            or compare_max_percent > 110
                            or compare_min_percent < 90
                            or compare_min_percent > 110
                            or late_intervals_ct_pc > 0
                        ):
                            interval_check = "NOK"
                        else:
                            interval_check = "OK"
                else:
                    icd_interval = "No ICD data found in VDE"
                    compare_mean_percent = "N/A"
                    compare_max_percent = "N/A"
                    compare_min_percent = "N/A"
                    interval_check = "N/A"
                    icd_interval_refactored = "N/A"

                if late_intervals_ct > 0 and lru_name not in selected_intervals:
                    selected_intervals[lru_name] = (
                        table,
                        intervals_np,
                        icd_interval,
                        mean_interval,
                        max_interval,
                        min_interval,
                        std_interval,
                    )

                interval_analysis.append(
                    {
                        "Influx Table": table,
                        "LRU Name": lru_name,
                        "Mean Interval (ms)": mean_interval,
                        "Max Interval (ms)": max_interval,
                        "Min Interval (ms)": min_interval,
                        "Std Dev Interval (ms)": std_interval,
                        "Coef. of Var. (%)": coefficient_of_variation,
                        "Max Z Score (std)": max_z_score,
                        "Min Z Score (std)": min_z_score,
                        "Intervals > 1.5x Mean (%)": late_intervals_ct_pc,
                        "Intervals > 1.5x Mean (count/total)": late_intervals_ratio,
                        "ICD Transmit Interval (ms)": icd_interval,
                        "Max Occurences": max_occurrences,
                        "ICD Int / Max Occ (ms)": icd_interval_refactored,
                        "Compare Mean (%)": compare_mean_percent,
                        "Compare Max (%)": compare_max_percent,
                        "Compare Min (%)": compare_min_percent,
                        "Interval Check": interval_check,
                    }
                )

    interval_analysis_df = pd.DataFrame(interval_analysis)

    return interval_analysis_df, selected_intervals


def plot_intervals_and_histograms(writer, selected_intervals, output_folder):
    """Plot intervals as both line plots and histograms with normal distribution overlays."""
    os.makedirs(output_folder, exist_ok=True)

    plot_count = 0
    sheet_count = 1
    max_plots_per_sheet = 9

    plot_files_line = []
    plot_files_histogram = []

    for lru_name, (
        table,
        intervals_np,
        icd_interval,
        mean_interval,
        max_interval,
        min_interval,
        std_interval,
    ) in selected_intervals.items():

        if pd.notna(icd_interval) and isinstance(icd_interval, (int, float)):
            icd_interval_str = f"{icd_interval} ms"
        else:
            "Not Available"
        title = f"Message: {table}, LRU: {
            lru_name}, ICD Interval: {icd_interval_str}"

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(
            range(len(intervals_np)),
            intervals_np,
            color="steelblue",
            linestyle="-",
            lw=1.3,
            label="Intervals",
        )
        ax.axhline(
            y=mean_interval,
            color="tomato",
            linestyle="--",
            lw=1.3,
            label="Mean Interval",
        )
        ax.set_title(title)
        ax.set_xlabel("Sample Number")
        ax.set_ylabel("Interval (ms)")
        ax.legend()

        plot_file_line = os.path.join(
            output_folder, f"line_plot_{plot_count + 1}.png")
        plt.savefig(plot_file_line, bbox_inches="tight")
        plt.close()
        plot_files_line.append(plot_file_line)

        fig, ax = plt.subplots(figsize=(15, 5))
        cnts, bins, ptches = ax.hist(
            intervals_np, bins=150, color="steelblue", label="Histogram"
        )
        bin_width = bins[1] - bins[0]
        scale_factor = len(intervals_np) * bin_width
        x = np.linspace(min_interval, max_interval, 300)
        p = norm.pdf(x, mean_interval, std_interval) * scale_factor
        ax.plot(x, p, color="green", linestyle=":",
                lw=1.3, label="Normal Distribution")
        ax.axvline(
            mean_interval, color="red", linestyle="--", lw=1.3, label="Mean Interval"
        )
        ax.axvline(
            min_interval, color="purple", linestyle="--", lw=1.3, label="Min Interval"
        )
        ax.axvline(
            max_interval, color="orange", linestyle="--", lw=1.3, label="Max Interval"
        )
        ax.set_title(title)
        ax.set_xlabel("Interval (ms)")
        ax.set_ylabel("Count")
        ax.legend()

        plot_file_histogram = os.path.join(
            output_folder, f"hist_plot_{plot_count + 1}.png"
        )
        plt.savefig(plot_file_histogram, bbox_inches="tight")
        plt.close()
        plot_files_histogram.append(plot_file_histogram)

        plot_count += 1

    plot_count = 0
    sheet_count = 1

    for i in range(0, len(plot_files_line), max_plots_per_sheet):
        sheet_name = f"Interval Plots {sheet_count}"
        worksheet = writer.book.add_worksheet(sheet_name)
        sheet_count += 1

        for j in range(max_plots_per_sheet):
            idx = i + j
            if idx >= len(plot_files_line):
                break

            worksheet.insert_image(j * 24, 0, plot_files_line[idx])
            worksheet.insert_image(j * 24, 20, plot_files_histogram[idx])

    if plot_count % max_plots_per_sheet != 0:
        worksheet.insert_image(
            (plot_count % max_plots_per_sheet) * 24, 0, plot_file_line
        )
        worksheet.insert_image(
            (plot_count % max_plots_per_sheet) * 24, 20, plot_file_histogram
        )


def export_to_excel(
    output_path,
    output_folder,
    summary_df,
    tables_w_data_df,
    tables_wo_data_df,
    missing_tables_df,
    icd_message_data_df,
    icd_param_data_df,
    interval_analysis_df,
    selected_intervals,
):
    """Export data to an Excel file with formatting."""
    excel.ExcelFormatter.header_style = None
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary",
                            index=False, header=False)
        interval_analysis_df.to_excel(
            writer,
            sheet_name="Interval Analysis",
            index=False,
            header=True,
            freeze_panes=(1, 0),
        )
        plot_intervals_and_histograms(
            writer, selected_intervals, f"{output_folder}/plots"
        )
        tables_w_data_df.to_excel(
            writer,
            sheet_name="Influx Tables With Data",
            index=False,
            header=True,
            freeze_panes=(1, 0),
        )
        tables_wo_data_df.to_excel(
            writer,
            sheet_name="Influx Tables Without Data",
            index=False,
            header=True,
            freeze_panes=(1, 0),
        )
        missing_tables_df.to_excel(
            writer,
            sheet_name="ICD Messages Not In Influx",
            index=False,
            header=True,
            freeze_panes=(1, 0),
        )
        icd_message_data_df.to_excel(
            writer,
            sheet_name="ICD Messages",
            index=False,
            header=True,
            freeze_panes=(1, 0),
        )
        icd_param_data_df.to_excel(
            writer,
            sheet_name="Name to dbTable Consistency",
            index=False,
            header=True,
            freeze_panes=(1, 0),
        )

        left_align_format = writer.book.add_format({"align": "left"})

        def format_sheet(sheet_name, df):
            """Format each sheet individually."""
            sheet = writer.sheets[sheet_name]
            column_range = f"A:{chr(64 + len(df.columns))}"
            sheet.set_column(column_range, None, left_align_format)
            sheet.autofit()
            if sheet_name != "Summary":
                sheet.autofilter(f"A1:{chr(64 + len(df.columns))}1")

        format_sheet("Summary", summary_df)
        format_sheet("Interval Analysis", interval_analysis_df)
        format_sheet("Influx Tables With Data", tables_w_data_df)
        format_sheet("Influx Tables Without Data", tables_wo_data_df)

    print(f"Analysis exported to Excel file '{output_path}'\n")


def main(args):
    """Main function to execute the script."""

    # Start timing
    script_start_time = time.time()

    # Parse CLI args
    credentials_path = args.credentials_path
    config = args.config
    start_time = args.start_time
    end_time = args.end_time
    load_pickle_path = args.load_pickle_path
    save_pickle_path = args.save_pickle_path

    # Load Influx credentials
    cert = read_certificate()
    influxdb_credentials = read_influxdb_credentials(credentials_path)
    host = influxdb_credentials["host"]
    token = influxdb_credentials["token"]
    database = influxdb_credentials["database"]

    # Load data stored in pickle, if provided
    if load_pickle_path:
        (
            tables_w_data_dict,
            tables_w_data_list,
            tables_wo_data_list,
            start_time,
            end_time,
            config,
            database,
            pickle_available,
        ) = load_pickle(load_pickle_path)
        if not pickle_available:
            print("\nData will be queried from InfluxDB:")
            pickle_available = False
    else:
        print("\nData will be queried from InfluxDB:")
        pickle_available = False

    # Summarise query
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Config: {config}")
    print(f"Database: {database}")

    # Run query, if no data already provided
    if not pickle_available:
        print("\nCommencing query")
        client = create_influxdb_client(token, host, database, cert)
        found_tables_list = query_influx_measurements(client)
        print(f"Found {len(found_tables_list)} tables\n")

        # Select threading method; no threading, map, or futures
        mode_dict = {1: "normal", 2: "tpemaprc", 3: "tpefutr"}
        mode = 2
        fallback_mode = 1
        function_name = f"query_influx_timestamps_{mode_dict[mode]}"

        # Run query
        try:
            func = globals()[function_name]
            (tables_w_data_dict, tables_w_data_list, tables_wo_data_list) = func(
                client, found_tables_list, start_time, end_time
            )
        except KeyError:
            print(
                f"Function {function_name} failed. Falling back to mode {
                    fallback_mode}."
            )
            fallback_function_name = (
                f"query_influx_timestamps_{mode_dict[fallback_mode]}"
            )
            func = globals()[fallback_function_name]
            (tables_w_data_dict, tables_w_data_list, tables_wo_data_list) = func(
                client, found_tables_list, start_time, end_time
            )

        # Save to pickle, if path provided at CLI
        if save_pickle_path:
            save_to_pickle(
                save_pickle_path,
                tables_w_data_dict,
                tables_w_data_list,
                tables_wo_data_list,
                start_time,
                end_time,
                config,
                database,
            )

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"output_verify_influx/{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(
        output_folder, f"output_verify_influx_{timestamp}.xlsx")
    working_dir = os.getcwd()

    # Summarise
    total_tables_w_data = len(tables_w_data_list)
    total_tables_wo_data = len(tables_wo_data_list)
    total_tables = total_tables_w_data + total_tables_wo_data

    summary_df = pd.DataFrame(
        {
            "Description": [
                "Influx Configuration File Path",
                "Configuration",
                "Query Start Time",
                "Query End Time",
                "Load Pickle File Path",
                "Save Pickle File Path",
                "Output File Path",
                "Working Directory",
                "Name to dbTable Consistency Status",
                "Total Tables",
                "Tables With Data",
                "Tables Without Data",
                "Tables (by LRU) With OK Intervals",
                "Tables (by LRU) With NOK Intervals",
                "Tables (by LRU) Not Analysed (N/A)",
            ],
            "Details": [
                credentials_path if pickle_available else "Credentials not used.",
                config,
                start_time,
                end_time,
                load_pickle_path if load_pickle_path else "No pickle used",
                save_pickle_path if save_pickle_path else "No output pickle requested",
                output_path,
                working_dir,
                total_tables,
                total_tables_w_data,
                total_tables_wo_data,
            ],
        }
    )

    tables_w_data_df = pd.DataFrame(
        tables_w_data_list, columns=["Influx Table"])
    tables_wo_data_df = pd.DataFrame(
        tables_wo_data_list, columns=["Influx Table"])

    # Export results to Excel
    export_to_excel(
        output_path,
        output_folder,
        summary_df,
        tables_w_data_df,
        tables_wo_data_df,
    )

    time_taken = math.floor(time.time() - script_start_time)
    global exception_counter
    print(f"Number of exceptions raised: {exception_counter}")
    print(f"Script run time = {time_taken} seconds\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Influx data with ICD definitions and check for gaps."
    )
    parser.add_argument(
        "credentials_path", help="Path to the InfluxDB credentials file"
    )
    parser.add_argument("config", help="Aircraft configuration")
    parser.add_argument(
        "start_time", help="Start time for the data query (ISO 8601 format)"
    )
    parser.add_argument(
        "end_time", help="End time for the data query (ISO 8601 format)"
    )
    parser.add_argument(
        "--load_pickle_path",
        required=False,
        help="Path to the pickle file for loading data",
    )
    parser.add_argument(
        "--save_pickle_path",
        required=False,
        help="Path to save the pickle file for caching data",
    )

    args = parser.parse_args()
    main(args)
