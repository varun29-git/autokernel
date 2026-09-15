import pytest

from autokernel.cli import demo_campaign
from autokernel.store import Store


@pytest.fixture
def campaign():
    return demo_campaign()


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "store")
    yield value
    value.close()
