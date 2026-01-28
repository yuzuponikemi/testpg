"""
Pytest configuration for async tests
"""

import pytest

# pytest-asyncioの設定
pytest_plugins = ('pytest_asyncio',)


def pytest_configure(config):
    """pytestの設定"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an asyncio test."
    )
