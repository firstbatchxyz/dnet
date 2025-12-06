import pytest


def pytest_addoption(parser):
    """Add --start-servers option to pytest."""
    parser.addoption(
        "--start-servers",
        action="store_true",
        default=False,
        help="Start dnet-api and dnet-shard servers automatically",
    )


@pytest.fixture(scope="module")
def start_servers_flag(request):
    """Get the --start-servers flag value."""
    return request.config.getoption("--start-servers")
