"""Module to handle fits files."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from astropy.io import fits
from astropy.table import Table

logger = logging.getLogger(__name__)


def get_header(
    file: str | Path,
    hdu: int = 0,
) -> dict[str, Any]:
    """Get the primary header of a fits file.

    Parameters
    ----------
    file : str or Path
        Path to the fits file.
    hdu : int, optional
        HDU number to read.
        Default is 0, which corresponds to the primary HDU.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the primary header.

    """
    file = Path(file)
    if not file.exists():
        message = f"File {file} does not exist."
        logger.error(message)
        raise FileNotFoundError(message)

    with fits.open(file) as hdul:  # type: ignore[no-untyped-call]
        base_hdu: fits.PrimaryHDU = hdul[hdu]  # pyright: ignore[reportAssignmentType]
        header = dict(base_hdu.header)

    logger.debug("Successfully read primary header from: %s", file)
    return header


def get_table(
    file: str | Path,
    hdu: int,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Get a table from a fits file and return it as a pandas.DataFrame.

    If the fits file contains a PrimaryHDU, the columns argument is ignored and
    all columns are read. If the fits file contains a TableHDU, only the
    specified columns are read. If a column contains multi-dimensional data, it
    is split into multiple columns in the resulting dataframe.

    Parameters
    ----------
    file : str or Path
        Path to the fits file.
    hdu : int
        HDU table number to read.
    columns : list[str], optional
        List of columns to read from the fits file.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing the data from the fits file.

    """
    file = Path(file)
    if not file.exists():
        message = f"File {file} does not exist."
        logger.error(message)
        raise FileNotFoundError(message)

    with fits.open(file) as hdul:  # type: ignore[no-untyped-call]
        base_hdu: fits.TableHDU = hdul[hdu]  # type: ignore[attr-defined]

        if isinstance(base_hdu, fits.PrimaryHDU):
            if columns is not None:
                logger.warning("Columns argument is ignored for PrimaryHDU.")
            table = Table(base_hdu.data)
            dataframe = table.to_pandas()
        else:
            if columns is None:
                columns = base_hdu.columns.names  # type: ignore[attr-defined]

            table = Table(base_hdu.data)
            table.keep_columns(columns)

            # Check for multi-dimensional columns and split them into multiple
            # columns if necessary
            column_names = table.columns.copy()
            for col in column_names:
                if len(table[col].shape) > 1:  # type: ignore[attr-defined]
                    logger.warning("Column %s has shape %s", col, table[col].shape)  # type: ignore[attr-defined]
                    logger.warning("Splitting the column into multiple columns.")
                    for i in range(table[col].shape[1]):  # type: ignore[attr-defined]
                        table.add_column(table[col][:, i], name=f"{col}_{i}", index=-1)  # type: ignore[attr-defined]
                    table.remove_column(col)  # type: ignore[attr-defined]

            dataframe = table.to_pandas()

    logger.debug("Successfully read dataframe from: %s", file)
    return dataframe
