"""Logging configuration."""

import logging


def configure_logging() -> None:
    """Send ERROR records to error.log in the current folder (rotation and UTC come with Phase 3)."""
    logging.basicConfig(
        filename='error.log',
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
