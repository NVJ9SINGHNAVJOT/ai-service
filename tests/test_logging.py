from __future__ import annotations

import logging


def test_setup_logging_quiets_noisy_dependency_loggers():
    """Network-heavy library INFO logs should not disrupt CLI progress output."""
    from app.core.logging import setup_logging

    setup_logging()

    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING
    assert logging.getLogger("huggingface_hub").level == logging.WARNING


def test_setup_logging_keeps_app_logging_level_configurable():
    """Application loggers should still follow the configured root level."""
    from app.core.logging import setup_logging

    setup_logging(level=logging.DEBUG)

    assert logging.getLogger().level == logging.DEBUG
