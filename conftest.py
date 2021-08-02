"""Pytest configuration."""
import numpy
import pytest


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    """Add numpy to namespace."""
    doctest_namespace["np"] = numpy
