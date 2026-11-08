"""Release helpers used by .github/workflows/release.yml.

    python scripts/release.py next-version [--write]   # print (and stamp) the next version
    python scripts/release.py check-wheel dist/*.whl   # refuse wheels that pollute site-packages

Versions are MAJOR.MINOR.PATCH with each part capped at 10.10.20. A release bumps
the patch of the latest released version, carrying into minor / major at the cap
(1.1.20 -> 1.2.0, 1.10.20 -> 2.0.0); 10.10.20 is the last possible release. The
version in pyproject.toml is a floor: set it above the latest release (e.g. 2.0.0)
to jump straight to that version.

Released versions are read from PyPI and from `v*` git tags (tags are pushed
right after each upload, so they cover PyPI's index lagging a few minutes).
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional

MAX_VERSION = (10, 10, 20)
VERSION_RE = re.compile(r'^v?(\d+)\.(\d+)\.(\d+)$')
type Version = tuple[int, int, int]


class VersionLimitError(ValueError):
    """The version is past MAX_VERSION, or no version is left to release."""


def parse(version: str) -> Optional[Version]:
    """'1.2.3' or 'v1.2.3' -> (1, 2, 3); anything else (pre-releases, etc.) -> None."""
    m = VERSION_RE.match(version.strip())
    return (int(m[1]), int(m[2]), int(m[3])) if m else None


def fmt(version: Version) -> str:
    return '.'.join(map(str, version))


def check_limits(version: Version) -> Version:
    for part, cap, name in zip(version, MAX_VERSION, ('major', 'minor', 'patch')):
        if part > cap:
            raise VersionLimitError(f"{fmt(version)}: {name} {part} is over the max of {cap} "
                                    f"(max version {fmt(MAX_VERSION)})")
    return version


def bump(version: Version) -> Version:
    """Next version: patch + 1, carrying into minor and major at the caps."""
    major, minor, patch = check_limits(version)
    patch += 1
    if patch > MAX_VERSION[2]:
        patch, minor = 0, minor + 1
    if minor > MAX_VERSION[1]:
        minor, major = 0, major + 1
    if major > MAX_VERSION[0]:
        raise VersionLimitError(f"{fmt(version)} is the max version {fmt(MAX_VERSION)}; "
                                "nothing left to release")
    return major, minor, patch


def next_version(floor: str, released: Iterable[str]) -> str:
    """The pyproject version if it is above every release, else bump the latest."""
    base = parse(floor)
    if base is None:
        raise ValueError(f"pyproject version {floor!r} is not MAJOR.MINOR.PATCH")
    check_limits(base)
    latest = max(filter(None, map(parse, released)), default=None)
    if latest is None or base > latest:
        return fmt(base)
    return fmt(bump(latest))


# ------- sources of released versions ------- #
def pypi_versions(package: str, index: str = 'https://pypi.org/pypi') -> list[str]:
    """Every version on the index; [] if the package hasn't been published yet."""
    req = urllib.request.Request(f"{index}/{package}/json",
                                 headers={'User-Agent': 'dragon-torch-release'})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return list(json.load(resp).get('releases', {}))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise


def tag_versions(repo: Path = Path('.')) -> list[str]:
    out = subprocess.run(['git', 'tag', '--list', 'v*'], cwd=repo, check=True,
                         capture_output=True, text=True).stdout
    return out.split()


# ------- pyproject ------- #
PROJECT_VERSION_RE = re.compile(r'(?ms)(^\[project\]\s*$.*?^version\s*=\s*)"([^"]*)"')


def read_pyproject(path: Path) -> tuple[str, str]:
    """(name, version) from the [project] table."""
    import tomllib
    project = tomllib.loads(path.read_text())['project']
    return project['name'], project['version']


def write_pyproject_version(path: Path, version: str) -> None:
    text = path.read_text()
    new, n = PROJECT_VERSION_RE.subn(lambda m: f'{m[1]}"{version}"', text, count=1)
    if n != 1:
        raise ValueError(f"no [project] version line found in {path}")
    path.write_text(new)


# ------- wheel layout ------- #
def check_wheel(path: Path, allowed: Iterable[str]) -> list[str]:
    """Top-level entries of the wheel outside `allowed` packages / modules.
    Anything else is installed straight into site-packages (e.g. a top-level
    `utils` or `tests`), clobbering other distributions."""
    allowed = set(allowed)
    problems = set()
    for name in zipfile.ZipFile(path).namelist():
        top = name.removeprefix('./').split('/', 1)[0]
        if top.endswith(('.dist-info', '.data')):
            continue
        module = top[:-3] if top.endswith('.py') else top
        if module not in allowed:
            problems.add(top)
    return sorted(problems)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    nv = sub.add_parser('next-version', help='print the next release version')
    nv.add_argument('--pyproject', type=Path, default=Path('pyproject.toml'))
    nv.add_argument('--index', default='https://pypi.org/pypi')
    nv.add_argument('--write', action='store_true', help='stamp the version into pyproject.toml')

    cw = sub.add_parser('check-wheel', help='fail if a wheel installs stray top-level files')
    cw.add_argument('wheels', nargs='+', type=Path)
    cw.add_argument('--allow', nargs='+', default=None,
                    help='allowed top-level packages (default: the import name of the project)')
    cw.add_argument('--pyproject', type=Path, default=Path('pyproject.toml'))

    args = p.parse_args(argv)
    if args.cmd == 'next-version':
        name, floor = read_pyproject(args.pyproject)
        released = pypi_versions(name, args.index) + tag_versions(args.pyproject.parent)
        version = next_version(floor, released)
        if args.write:
            write_pyproject_version(args.pyproject, version)
        print(version)
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f:
                f.write(f"version={version}\n")
        return 0

    allowed = args.allow or [read_pyproject(args.pyproject)[0].replace('-', '_')]
    failed = False
    for wheel in args.wheels:
        problems = check_wheel(wheel, allowed)
        if problems:
            failed = True
            print(f"{wheel.name}: installs {problems} into site-packages; only {allowed} "
                  "(and *.dist-info) are allowed", file=sys.stderr)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
