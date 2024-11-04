import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import matplotlib.gridspec as gridspec
import os
from datetime import datetime


def save_to_parquet(df, file_path):
    """Save DataFrame to a Parquet file."""
    try:
        print("\nSaving to parquet")
        df.to_parquet(file_path, index=False)
        print(f"DataFrame successfully saved to '{file_path}'\n")
    except Exception as e:
        print(f"Error saving DataFrame to Parquet: {e}\n")
        raise


def save_to_csv(df, file_path, index):
    """Save DataFrame to a CSV file."""
    try:
        print("\nSaving to CSV")
        df.to_csv(file_path, index=index)
        print(f"DataFrame successfully saved to '{file_path}'")
    except Exception as e:
        print(f"Error saving DataFrame to CSV: {e}")
        raise


def detect_overloads_per_segment(df, segment_ids, params, threshold=40):
    """Function to detect overloads for each parameter based on segments."""

    # Create initial overload columns for each parameter if they exist in the DataFrame
    for param in params:
        if param in df.columns:
            df[f"overload-{param}"] = False

    # Populate overload columns
    unique_segments = segment_ids.unique()
    for segment_index in unique_segments:
        segment_data = df[segment_ids == segment_index]
        if segment_data.empty:
            continue

        for param in params:
            overload_col = f"overload-{param}"
            if param in segment_data.columns:
                param_data = segment_data[param]
                if param_data.notna().any():
                    mean_value = np.mean(param_data.abs())
                    print(f"Segment: {segment_index}, Param: {
                          param}, Mean: {round(mean_value, 2)}")
                    # Set overloads based on threshold
                    df.loc[segment_data.index,
                           overload_col] = param_data.abs() > threshold

    # Print overload counts for each parameter
    for param in params:
        overload_col = f"overload-{param}"
        if overload_col in df.columns:
            count_overloads = df[overload_col].sum()
            total_records = len(df)
            print(f"{overload_col}: {count_overloads} / {total_records}")

    return df


def plot_time_series(ax_time, segment_data, param_data, param, segment_index):
    """Plot time series data."""
    ax_time.plot(
        segment_data.time,
        param_data,
        linestyle="-",
        linewidth=0.8,
        label=param,
    )
    ax_time.set_xlabel("Time")
    ax_time.set_ylabel(param)
    ax_time.set_title(f"Segment {segment_index}: {param} Time Series")
    ax_time.grid(True)
    ax_time.xaxis.set_major_formatter(
        mdates.DateFormatter("%d.%m.%y-%H:%M:%S"))
    plt.setp(ax_time.get_xticklabels(), rotation=45, ha="right")


def plot_overloads(ax_time, overload_col, segment_data):
    """Add red blocks for overload samples."""
    time_interval = 0.5
    if overload_col in segment_data.columns:
        overload_indices = segment_data["time"][segment_data[overload_col]]
        if not overload_indices.empty:
            start_time = overload_indices.iloc[0]
            end_time = start_time + pd.Timedelta(seconds=time_interval)
            ax_time.axvspan(start_time, end_time, color="red", alpha=0.6)
            for overload_index in overload_indices:
                if overload_index > end_time:
                    start_time = overload_index
                    end_time = start_time + pd.Timedelta(seconds=time_interval)
                    ax_time.axvspan(start_time, end_time,
                                    color="red", alpha=0.6)
    else:
        print(f"Overload column '{overload_col}' not found in segment data")


def create_segments(df):
    """Create segments of continuous data."""

    # Convert the 'time' column to datetime and set it as the index
    df["time"] = pd.to_datetime(df["time"])

    # Info
    print(f"Start of database: {df.time.min()}")
    print(f"End of database: {df.time.max()}")
    print(f"Length of database: {len(df)}")

    # Calculate time differences
    time_diff = df.time.diff().dt.total_seconds()

    # Identify segments of continuous data and adjust segment IDs to start from 1
    gap = 0.1
    segment_ids = (time_diff > gap).cumsum() + 1

    return df, segment_ids


def plot_segments(df, segment_ids, params_to_plot, output_folder):
    """Plot segments as time series with overloads, and fft."""

    def plot_empty_time_series():
        """Plot an empty time series."""
        ax_time.plot([], [], linestyle="-", linewidth=0.8, label=param)
        ax_time.set_xlabel("Time")
        ax_time.set_ylabel(param)
        ax_time.set_title(f"Segment {segment_index}: {
                          param} Time Series (No Data)")
        ax_time.grid(True)
        ax_time.xaxis.set_major_formatter(
            mdates.DateFormatter("%d.%m.%y-%H:%M:%S"))
        plt.setp(ax_time.get_xticklabels(), rotation=45, ha="right")

    def plot_empty_fft():
        """Plot an empty FFT."""
        ax_fft.plot([], [], "r--")
        ax_fft.set_xlabel("Frequency [Hz]")
        ax_fft.set_ylabel("FFT Magnitude")
        ax_fft.set_title(f"Segment {segment_index}: {param} FFT (No Data)")
        ax_fft.grid(True)

    # Initialize lists to track good and skipped segments
    good_segments = []
    skipped_segments = []

    # Iterate over each adjusted segment and generate plots
    unique_segments = segment_ids.unique()
    for segment_index in unique_segments:

        # Get data for the current segment
        segment_data = df[segment_ids == segment_index]

        # Check if segment_data empty
        if segment_data.empty:
            print(
                f"Segment {segment_index} is empty after filtering. Skipping...")
            skipped_segments.append(segment_index)
            continue

        # Prepare plotting
        fig = plt.figure(figsize=(30, 60))
        gs = gridspec.GridSpec(9, 2, width_ratios=[
                               2, 1], height_ratios=[1] * 9)

        # Plot each param in turn, times series and fft
        for i, param in enumerate(params_to_plot):

            # Overoad of interest
            overload_col = f"overload-{param}"

            # Build subplots
            ax_time = fig.add_subplot(gs[i, 0])
            ax_fft = fig.add_subplot(gs[i, 1])

            # Begin plotting
            if param in segment_data.columns:
                param_data = segment_data[param]

                # Time series plot
                plot_time_series(ax_time, segment_data,
                                 param_data, param, segment_index)

                # Plot overloads
                plot_overloads(ax_time, overload_col, segment_data)

                # FFT calculation for the segment
                if param_data.notna().any():
                    signal = param_data.dropna()
                    N = len(signal)
                    if N > 1:
                        segment_time_diff = segment_data.time.diff().dt.total_seconds()
                        mean_interval = segment_time_diff[1:].mean()
                        sampling_rate = 1 / mean_interval
                        print(
                            f"Segment: {segment_index}, Param: {
                                param}, Interval: "
                            f"{round(mean_interval, 6)}, Rate: {
                                round(sampling_rate, 3)}"
                        )
                        fft_values = np.fft.fft(signal)
                        fft_freqs = np.fft.fftfreq(N, mean_interval)

                        # Normalize the FFT output
                        fft_values = fft_values / len(signal)

                        # Plot the magnitude of the FFT
                        ax_fft.plot(
                            fft_freqs[: N // 2],
                            np.abs(fft_values[: N // 2]),
                            "r--",
                            linewidth=0.8,
                        )
                        ax_fft.set_xlabel("Frequency [Hz]")
                        ax_fft.set_ylabel("FFT Magnitude")
                        ax_fft.set_title(
                            f"Segment {segment_index}: {param} FFT")
                        ax_fft.grid(True)

                        # Set appropriate x-axis limits
                        ax_fft.set_xlim(0, sampling_rate / 2)
                    else:
                        plot_empty_fft()
                else:
                    plot_empty_fft()
            else:
                plot_empty_time_series()
                plot_empty_fft()

        plt.tight_layout()
        output_file = f"{output_folder}/segment_{segment_index}_plot.png"
        plt.savefig(output_file)
        plt.close()
        good_segments.append(segment_index)
        print(f"Saved plot for segment {segment_index} to '{output_file}'")

    print(f"Total segments processed: {len(unique_segments)}")
    print(f"Successfully processed segments: {len(good_segments)}")
    print(f"Skipped segments: {len(skipped_segments)}")


def main():

    # Define the path to the Parquet file
    parquet_path = (
        "output_param_query/20240921_101400/" "output_param_query_combined_20240921_101400.parquet"
    )

    # Load the Parquet file
    print(f"\nLoading parquet file: '{parquet_path}'")
    df = pd.read_parquet(parquet_path)

    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"output_overloads/filtered_segments_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    # Create plot segments
    df, segment_ids = create_segments(df)

    # List parameters to plot
    params_to_plot = [
        "param1",
        "param2",
    ]
    params_to_plot = [p.lower() for p in params_to_plot]

    # Detect overloads
    df = detect_overloads_per_segment(df, segment_ids, params_to_plot)

    # Plot segments
    plot_segments(df, segment_ids, params_to_plot, output_folder)

    # Drop columns, if any
    columns_to_drop = [
        "param3",
        "param4",
    ]
    df.drop(columns=columns_to_drop, inplace=True)

    # Save the updated dataframe to a new parquet file
    output_parquet_path = f"{output_folder}/overload_labels.parquet"
    save_to_parquet(df, output_parquet_path)

    # Ouputs head(100) to csv
    output_parquet_path = f"{output_folder}/overload_labels_head.csv"
    save_to_csv(df.head(100), output_parquet_path, False)

    # Output describe() to csv
    output_parquet_path = f"{output_folder}/overload_labels_describe.csv"
    save_to_csv(df.describe(), output_parquet_path, True)

    # Output isnulls.sum() to csv
    output_parquet_path = f"{output_folder}/overload_labels_isnull.csv"
    save_to_csv(df.isnull().sum(), output_parquet_path, True)

    print(f"Script execution completed. Results saved to '{output_folder}'.")


if __name__ == "__main__":
    main()
