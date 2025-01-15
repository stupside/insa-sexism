import csv

import io

from ast import literal_eval

from rich.progress import track


def read(
    file: io.TextIOWrapper,
):
    for row in track(csv.DictReader(file), description="Loading data"):

        for key, value in row.items():
            if value.startswith("[") and value.endswith("]"):
                try:
                    row[key] = literal_eval(value)
                except (ValueError, SyntaxError):
                    pass  # Keep original string if parsing fails

        yield row
