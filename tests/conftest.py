import pytest
import requests

SEMASIA_CELEBA_REPO = 'spaicom-lab/semasia-celeba'
# smallest embedding model in the repo, keeps streamed rows light
SEMASIA_CELEBA_MODEL = 'bat_resnext26ts.ch_in1k'
SEMASIA_CELEBA_EMBED_DIM = 2048


def pytest_configure(config):
    config.addinivalue_line(
        'markers', "network: hits the Hugging Face Hub (deselect with -m 'not network')")


@pytest.fixture(scope='session')
def hf_hub_online():
    try:
        requests.head('https://huggingface.co', timeout=5)
    except requests.RequestException:
        pytest.skip('Hugging Face Hub unreachable')
