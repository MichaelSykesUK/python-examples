import datetime
import argparse
import pandas as pd
from typing import Tuple, Dict, List
import logging
from pandas.io.formats import excel


def setup_logging() -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        filename="iads_config_to_excel.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def read_iads_file(input_file: str) -> List[str]:
    """Reads an IADS config file and returns the content as a list of lines."""
    with open(input_file, 'r') as infile:
        return infile.readlines()


def parse_table_definition(line: str) -> List[str]:
    """Extracts column headers from the table definition line."""
    line = line.strip()[16:]  # Skip "TableDefinition "
    return ['TableDefinition'] + line.split("|")


def process_table_lines(lines: List[str]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Processes lines of an IADS config file to extract tables and their data."""
    dfs_all: Dict[str, pd.DataFrame] = {}
    found_tables: List[str] = []
    table_data: List[List[str]] = []
    column_headers: List[str] = []
    current_table_name: str = ""
    found_section = False

    for line in lines:
        if not found_section:
            if line.startswith("   Table "):
                found_section = True
                current_table_name = line[9:].strip()
                found_tables.append(current_table_name)
                print(f"Processing IADS table '{current_table_name}'...")
        elif line.startswith("   }"):
            # End of table
            dfs_all[current_table_name] = pd.DataFrame(table_data, columns=column_headers)
            table_data = []
            column_headers = []
            found_section = False
        elif line.startswith("      TableDefinition "):
            # Extract column headers
            column_headers = parse_table_definition(line)
        elif line and line.strip().split()[0] not in ("{", "}", "TableContents", "TableType"):
            # Process data rows
            column_1_data, remaining_data = line.strip().split(' ', 1)
            all_other_column_data = remaining_data.split('|')
            row_data = [column_1_data] + all_other_column_data
            table_data.append(row_data)

    return dfs_all, found_tables


def iads_df_to_excel(dfs_all: Dict[str, pd.DataFrame], found_tables: List[str]) -> None:
    """Outputs a dictionary of DataFrames to an Excel file."""
    current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file_name = f"iads_config_to_excel_{current_datetime}.xlsx"
    print(f"Creating output file '{output_file_name}'...")
    excel.ExcelFormatter.header_style = None

    try:
        with pd.ExcelWriter(output_file_name, engine="xlsxwriter") as writer:
            for table_name in found_tables:
                print(f"Creating sheet '{table_name}' in output file...")
                df = dfs_all[table_name]
                df.to_excel(
                    writer,
                    sheet_name=table_name[:31],
                    index=False,
                    header=True,
                    freeze_panes=(1, 0),
                )
                sheet = writer.sheets[table_name[:31]]
                num_columns = len(df.columns)
                left_align_format = writer.book.add_format({"align": "left"})
                sheet.set_column(0, num_columns - 1, None, left_align_format)
                sheet.autofilter(0, 0, 0, num_columns - 1)
                sheet.autofit()
        print(f"Output file '{output_file_name}' is complete.")
    except Exception as e:
        logging.error(f"Error creating output file '{output_file_name}'. "
                      f"Check log for details: {e}")


def main() -> None:
    """Run the main function of the script."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="This script outputs a single IADS config file to Excel"
    )
    parser.add_argument("input", help="Path to single IADS config file")
    args = parser.parse_args()

    try:
        lines = read_iads_file(args.input)
        dfs_all, found_tables = process_table_lines(lines)
        iads_df_to_excel(dfs_all, found_tables)
    except Exception as e:
        logging.error(f"Script failed. Check log file 'iads_config_to_excel.log' for details: {e}")


if __name__ == "__main__":
    main()
