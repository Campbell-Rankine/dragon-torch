from pydantic import BaseModel
from typing import Union, Literal, Annotated
import os

"""Base data-loading primitives: pydantic configs, pluggable sources, base dataset.
 
Suggested layout once this grows:
    mylib/data/config.py    -> configs
    mylib/data/sources.py   -> Source ABC, registry, DiskSource, HTTPSource
    mylib/data/datasets.py  -> BaseDataset
    mylib/data/loaders.py   -> build_dataloader
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar
from .config import DiskSourceConfig, HTTPSourceConfig


def build_source(cfg: DiskSourceConfig | HTTPSourceConfig) -> Source:
    return Source._registry[cfg.kind](cfg)

# --------------------------------------------------------------------------- #
# Sources: "where do bytes come from". Dataset code never knows which one.
# --------------------------------------------------------------------------- #


class Source(ABC):
    kind: ClassVar[str]
    _registry: ClassVar[dict[str, type["Source"]]] = {}

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if hasattr(cls, "kind"):
            Source._registry[cls.kind] = cls

    @abstractmethod
    def list_keys(self) -> list[str]: ...

    @abstractmethod
    def read(self, key: str) -> bytes: ...


class DiskSource(Source):
    kind = "disk"

    def __init__(self, cfg: DiskSourceConfig):
        self.cfg = cfg

    def list_keys(self) -> list[str]:
        exts = {e.lower()
                for e in self.cfg.extensions} if self.cfg.extensions else None
        paths = (p for p in self.cfg.root.glob(
            self.cfg.pattern) if p.is_file())
        return sorted(
            str(p.relative_to(self.cfg.root))
            for p in paths
            if exts is None or p.suffix.lower() in exts
        )

    def read(self, key: str) -> bytes:
        path = (self.cfg.root / key).resolve()
        if self.cfg.root not in path.parents:  # block path traversal
            raise ValueError(f"key escapes root: {key}")
        return path.read_bytes()


class HTTPSource(Source):
    kind = "http"

    def __init__(self, cfg: HTTPSourceConfig):
        self.cfg = cfg
        self._session = None  # created lazily, once per worker process

    @property
    def session(self):
        if self._session is None:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            s = requests.Session()
            retry = Retry(
                total=self.cfg.max_retries,
                backoff_factor=self.cfg.backoff_factor,
                status_forcelist=(429, 500, 502, 503, 504),
            )
            s.mount("https://", HTTPAdapter(max_retries=retry))
            s.mount("http://", HTTPAdapter(max_retries=retry))
            if self.cfg.auth_token:
                s.headers["Authorization"] = f"Bearer {self.cfg.auth_token.get_secret_value()}"
            self._session = s
        return self._session

    # sessions aren't picklable (spawn workers)
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_session"] = None
        return state

    def list_keys(self) -> list[str]:
        return list(self.cfg.keys)

    def read(self, key: str) -> bytes:
        r = self.session.get(
            f"{self.cfg.base_url}/{key.lstrip('/')}", timeout=self.cfg.timeout_s)
        r.raise_for_status()
        return r.content
