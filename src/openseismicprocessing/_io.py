import segyio
import pandas as pd
import re

def open_segy_data(filePath, ignore_geometry = True):
    return segyio.open(filePath, "r", ignore_geometry=ignore_geometry)

def parse_trace_headers(segyfile):
    '''
    Parse the segy file trace headers into a pandas dataframe.
    Column names are defined from segyio internal tracefield
    One row per trace
    '''
    # Get all header keys
    n_traces = segyfile.tracecount
    headers = segyio.tracefield.keys
    # Initialize dataframe with trace id as index and headers as columns
    df = pd.DataFrame(index=range(0, n_traces),
                      columns=headers.keys())
    # Fill dataframe with all header values
    for k, v in headers.items():
        df[k] = segyfile.attributes(v)[:]
    return df

def parse_text_header(segyfile):
    """
    Format SEGY text header into a clean, readable dictionary.
    """
    try:
        # Wrap the raw header text to standard 80-character lines
        raw_header = segyio.tools.wrap(segyfile.text[0])
    except Exception as e:
        print(f"❌ Error wrapping text header: {e}")
        return None

    # Split header based on the expected pattern; adjust if necessary.
    parts = re.split(r'C ', raw_header)
    if len(parts) < 2:
        print("❌ Error: Unexpected header format.")
        return None

    # The first split element might be empty, so we take the rest
    cut_header = parts[1:]

    # Replace newline characters with a space and strip extra whitespace
    text_header = [line.replace('\n', ' ').strip() for line in cut_header]
    
    # Optionally adjust the last line if there are trailing unwanted characters
    if text_header and len(text_header[-1]) > 2:
        text_header[-1] = text_header[-1][:-2].strip()

    # Format into a dictionary with keys like "C01", "C02", etc.
    clean_header = {}
    for i, item in enumerate(text_header, start=1):
        key = "C" + str(i).rjust(2, '0')
        clean_header[key] = item

    return clean_header