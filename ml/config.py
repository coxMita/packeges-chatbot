"""Shared paths and constants for the ML pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
MALICIOUS_DIR = RAW / "malicious"
BENIGN_DIR = RAW / "benign"
CACHE = DATA / "cache"
PROCESSED = DATA / "processed"

MODELS = ROOT / "ml" / "models"
REPORTS = ROOT / "ml" / "reports"

FEATURES_PARQUET = PROCESSED / "features.parquet"

# --- upstream sources -------------------------------------------------------

DATADOG_REPO = "https://github.com/DataDog/malicious-software-packages-dataset.git"
DATADOG_SAMPLE_PASSWORD = b"infected"  # documented by the dataset authors

TOP_PYPI_URL = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages.min.json"
PYPI_SIMPLE_INDEX = "https://pypi.org/simple/"
PYPI_JSON_API = "https://pypi.org/pypi/{name}/json"

OSSF_MALICIOUS_REPO = "https://github.com/ossf/malicious-packages.git"

# --- benign sampling --------------------------------------------------------

# Popular packages: the realistic negative class.
N_TOP_PACKAGES = 5000

# Obscure packages sampled at random from the full PyPI index. These are the
# *hard negatives* -- small, amateur, low-effort but benign. Without them the
# model learns "small package == malicious" and reports a meaningless 99% F1.
N_RANDOM_PACKAGES = 3000

# Packages ranked below this are considered "obscure" enough to be a hard
# negative rather than a second copy of the popular class.
TOP_RANK_EXCLUSION = 15000

RANDOM_SEED = 1337

# --- extraction guards ------------------------------------------------------

# These are zip-bomb guards, NOT size filters. They are set generously on
# purpose: a cap tight enough to reject a real package (numpy's sdist has ~6000
# files) would quietly drop the largest benign projects and bias the benign
# class toward small packages -- reintroducing the exact "small == malicious"
# artifact that acquire_benign.py's hard-negative pool exists to prevent.
# Actual bombs expand by orders of magnitude more than this.
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_UNPACKED_BYTES = 2 * 1024 * 1024 * 1024
MAX_FILES_PER_PACKAGE = 60_000

DOWNLOAD_TIMEOUT = 30
HTTP_USER_AGENT = "packeges-chatbot-research/0.1 (+https://github.com/coxMita/packeges-chatbot)"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes into."""
    for d in (RAW, MALICIOUS_DIR, BENIGN_DIR, CACHE, PROCESSED, MODELS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
