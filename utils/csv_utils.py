from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


def save_csv(
    path: Path | str,
    header: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    overwrite: bool = True,
) -> None:
    """
    Persist tabular data into a CSV file at ``path``.

    Args:
        path: Destination file path.
        header: Column names written as the first row.
        rows: Iterable with the data rows.
        overwrite: When True, rewrite any existing file. When False the rows are
            appended and the header is only written if the file is empty.
    """
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if overwrite else "a"
    needs_header = overwrite or not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if needs_header:
            writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))
