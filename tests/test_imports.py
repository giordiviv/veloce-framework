"""Test library imports."""  # noqa: INP001


def test_import_velocefw() -> None:
    """Test import of velocefw."""
    import velocefw  # noqa: F401


def test_import_read_fits() -> None:
    """Test import of read_fits."""
    from velocefw import read_fits  # noqa: F401
    from velocefw.read_fits import get_header, get_table  # noqa: F401


def test_import_parametrization() -> None:
    """Test import of Parametrization."""
    from velocefw.model import Parametrization  # noqa: F401


def test_import_model() -> None:
    """Test import of velocefw.model."""
    from velocefw import model  # noqa: F401


def test_import_polynomial() -> None:
    """Test import of PolynomialBasis."""
    from velocefw.model import PolynomialBasis  # noqa: F401


def test_import_constant() -> None:
    """Test import of Constant."""
    from velocefw.model import Constant  # noqa: F401


def test_import_fourier_series() -> None:
    """Test import of FourierSeries."""
    from velocefw.model import FourierSeries, calculate_phase  # noqa: F401
