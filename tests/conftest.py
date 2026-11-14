import os
import tempfile
from pathlib import Path

# isolate glossary/pins/sessions from the developer's real data before anything imports config
os.environ.setdefault("AGENT_DATA_DIR", tempfile.mkdtemp(prefix="agent_test_"))

import pytest  # noqa: E402

from analyst_agent.data.store import Session  # noqa: E402
from analyst_agent.glossary import Glossary  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "sample_data" / "retail_orders.csv"


@pytest.fixture(scope="session")
def sample_session():
    s = Session()
    s.load_file(SAMPLE)
    return s


@pytest.fixture
def session():
    s = Session()
    s.load_file(SAMPLE)
    return s


@pytest.fixture
def glossary(tmp_path):
    return Glossary(tmp_path / "glossary.sqlite")
