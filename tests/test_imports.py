"""Test library imports."""  # noqa: INP001


def test_import_velocefw() -> None:
    """Test import of velocefw."""
    import velocefw  # noqa: F401


def test_import_model() -> None:
    """Test import of velocefw.model."""
    from velocefw import model  # noqa: F401


def test_import_polynomial() -> None:
    """Test import of PolynomialBasis."""
    from velocefw.model import PolynomialBasis  # noqa: F401
