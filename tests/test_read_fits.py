"""Test reading of FITS files."""  # noqa: INP001

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from velocefw.read_fits import get_header, get_table

logger = logging.getLogger(__name__)


def test_get_table_file_not_found() -> None:
    """Test get_table if file does not exist."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        get_table("non_existent_file.fits", hdu=1)


def test_get_header_file_not_found() -> None:
    """Test get_header if file does not exist."""
    with pytest.raises(FileNotFoundError, match="does not exist"):
        get_header("non_existent_file.fits")


def test_get_header_and_table() -> None:
    """Test get_header and get_table functions."""
    # Create a temporary FITS file with a simple header
    with tempfile.NamedTemporaryFile(suffix=".fits") as tmp:
        # Create a simple FITS file with a header
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["TESTKEY"] = "TESTVALUE"
        # Create three simple table HDUs
        col1 = fits.Column(name="id", format="I", array=[1, 2, 3])
        col2 = fits.Column(name="value", format="E", array=[1.1, 2.2, 3.3])
        table1 = fits.BinTableHDU.from_columns([col1, col2], name="TABLE1")

        col3 = fits.Column(name="flag", format="L", array=[True, False, True])
        col4 = fits.Column(name="name", format="10A", array=["A", "B", "C"])
        table2 = fits.BinTableHDU.from_columns([col3, col4], name="TABLE2")

        col5 = fits.Column(name="x", format="D", array=[0.1, 0.2, 0.3])
        col6 = fits.Column(
            name="y",
            format="3D",
            array=[
                np.array([0.4, 0.5, 0.6]),
                np.array([0.7, 0.8, 0.9]),
                np.array([1.0, 1.1, 1.2]),
            ],
        )
        table3 = fits.BinTableHDU.from_columns([col5, col6], name="TABLE3")
        hdul = fits.HDUList([primary_hdu, table1, table2, table3])
        hdul.writeto(tmp.name, overwrite=True)
        tmp_path = Path(tmp.name)

        # Test get_header
        header = get_header(tmp_path)
        if header.get("TESTKEY") != "TESTVALUE":
            msg = f"Header value for TESTKEY does not match expected value. {header}"
            logger.error(msg)
            raise RuntimeError(msg)

        # Test get_table
        table = get_table(tmp_path, hdu=1)
        if not isinstance(table, pd.DataFrame):
            msg = f"Returned object is of type {type(table)}, "
            msg += "expected pandas.DataFrame."
            logger.error(msg)
            raise TypeError(msg)
        if table.columns.to_list() != ["id", "value"]:
            msg = f"Table column names {table.columns.to_list()} do not match "
            msg += "expected ['id', 'value']."
            logger.error(msg)
            raise RuntimeError(msg)
        # Test on Primary HDU
        table = get_table(tmp_path, hdu=0)
        table = get_table(tmp_path, hdu=0, columns=["TESTKEY_wrong"])
        # Test specified columns that do not exist
        with pytest.raises(KeyError, match="does not exist"):
            get_table(tmp_path, hdu=1, columns=["non_existent_column"])
        # Test specified columns that do exist
        table = get_table(tmp_path, hdu=3, columns=["x"])
        if table.columns.to_list() != ["x"]:
            msg = f"Table column names {table.columns.to_list()} do not match "
            msg += "expected ['x']."
            logger.error(msg)
            raise RuntimeError(msg)
        # Test multi-dimensional columns are split correctly
        table = get_table(tmp_path, hdu=3)
        if table.columns.to_list() != ["x", "y_0", "y_1", "y_2"]:
            msg = f"Table column names {table.columns.to_list()} do not match "
            msg += "expected ['x', 'y_0', 'y_1', 'y_2']."
            logger.error(msg)
            raise RuntimeError(msg)
