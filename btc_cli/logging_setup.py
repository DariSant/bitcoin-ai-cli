"""Logging configuration."""

import logging

from btc_cli import config


def configure_logging() -> None:
    """Send ERROR records to error.log in the data folder (rotation and UTC come with Phase 3)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=config.ERROR_LOG,
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
