import importlib.util
import zipfile
from pathlib import Path
import pytest

_spec = importlib.util.spec_from_file_location(
    'release', Path(__file__).resolve().parents[1] / 'scripts' / 'release.py')
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)


class TestBump:
    @pytest.mark.parametrize('before,after', [
        ((1, 1, 0), (1, 1, 1)),
        ((1, 1, 19), (1, 1, 20)),
        ((1, 1, 20), (1, 2, 0)),      # patch cap carries into minor
        ((1, 10, 20), (2, 0, 0)),     # minor cap carries into major
        ((9, 10, 20), (10, 0, 0)),
        ((10, 10, 19), (10, 10, 20)),
    ])
    def test_carries(self, before, after):
        assert release.bump(before) == after

    def test_exhausted(self):
        with pytest.raises(release.VersionLimitError, match='nothing left'):
            release.bump((10, 10, 20))

    @pytest.mark.parametrize('bad', [(11, 0, 0), (1, 11, 0), (1, 1, 21)])
    def test_over_limit(self, bad):
        with pytest.raises(release.VersionLimitError, match='over the max'):
            release.bump(bad)


class TestNextVersion:
    def test_first_release_uses_pyproject(self):
        assert release.next_version('1.1.0', []) == '1.1.0'

    def test_bumps_latest_release(self):
        assert release.next_version('1.1.0', ['1.1.0', '1.1.3', 'v1.1.2']) == '1.1.4'

    def test_pyproject_floor_jumps_ahead(self):
        assert release.next_version('2.0.0', ['1.4.7']) == '2.0.0'

    def test_ignores_non_release_versions(self):
        assert release.next_version('1.0.0', ['1.0.0', '1.0.5rc1', 'nightly']) == '1.0.1'

    def test_tags_and_index_merge(self):
        # a tag pushed after an upload covers PyPI's index lagging behind
        assert release.next_version('1.0.0', ['1.0.0', 'v1.0.1']) == '1.0.2'

    def test_rejects_bad_pyproject_version(self):
        with pytest.raises(ValueError, match='MAJOR.MINOR.PATCH'):
            release.next_version('1.0', [])
        with pytest.raises(release.VersionLimitError):
            release.next_version('1.11.0', [])

    def test_parse(self):
        assert release.parse('v1.2.3') == (1, 2, 3) and release.parse('1.2') is None


def test_write_pyproject_version(tmp_path):
    path = tmp_path / 'pyproject.toml'
    path.write_text('[project]\nname = "dragon-torch"\nversion = "1.1.0"\n\n'
                    '[tool.other]\nversion = "9.9.9"\n')
    release.write_pyproject_version(path, '1.1.5')
    assert release.read_pyproject(path) == ('dragon-torch', '1.1.5')
    assert 'version = "9.9.9"' in path.read_text()  # only [project] changes


def wheel(tmp_path, names):
    path = tmp_path / 'pkg-1.0.0-py3-none-any.whl'
    with zipfile.ZipFile(path, 'w') as z:
        for n in names:
            z.writestr(n, '')
    return path


def test_check_wheel(tmp_path):
    good = wheel(tmp_path, ['dragon_torch/__init__.py', 'dragon_torch-1.0.0.dist-info/METADATA'])
    assert release.check_wheel(good, ['dragon_torch']) == []
    bad = wheel(tmp_path, ['./utils/__init__.py', './tests/x.py', 'dragon_torch/a.py', 'main.py'])
    assert release.check_wheel(bad, ['dragon_torch']) == ['main.py', 'tests', 'utils']


def test_cli_next_version(tmp_path, monkeypatch, capsys):
    (tmp_path / 'pyproject.toml').write_text('[project]\nname = "dragon-torch"\nversion = "1.1.0"\n')
    monkeypatch.setattr(release, 'pypi_versions', lambda name, index: ['1.1.0', '1.1.1'])
    monkeypatch.setattr(release, 'tag_versions', lambda repo: ['v1.1.2'])
    out_file = tmp_path / 'out'
    monkeypatch.setenv('GITHUB_OUTPUT', str(out_file))
    assert release.main(['next-version', '--pyproject', str(tmp_path / 'pyproject.toml'),
                         '--write']) == 0
    assert capsys.readouterr().out.strip() == '1.1.3'
    assert out_file.read_text() == 'version=1.1.3\n'
    assert release.read_pyproject(tmp_path / 'pyproject.toml')[1] == '1.1.3'
