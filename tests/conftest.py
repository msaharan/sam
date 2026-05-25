import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def fixture_prices() -> Path:
    path = FIXTURES / "ohlcv_ma_baseline.parquet"
    if not path.exists():
        from scripts.generate_fixtures import main

        main()
    return path
