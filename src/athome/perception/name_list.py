"""Editable list of detector names, kept in a JSON file: the fixed list of the
perception check, changed from its page. The default list stays as it is, so
the search's list (configs/perception) is not changed by a test."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence

MAX_NAME = 60


def clean(name: str) -> str:
    return " ".join(str(name).casefold().split())


class NameList:
    def __init__(self, path: Path, default: Sequence[str]):
        self.path = Path(path)
        self.default = list(dict.fromkeys(clean(n) for n in default))
        self.names: List[str] = list(self.default)
        if self.path.is_file():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.names = list(dict.fromkeys(clean(n) for n in saved["names"] if clean(n)))

    @property
    def is_default(self) -> bool:
        return self.names == self.default

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"names": self.names}, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, self.path)

    def add(self, name: str) -> str:
        """The error for the page, '' when added or already there."""
        name = clean(name)
        if not name:
            return "이름이 비어 있음"
        if len(name) > MAX_NAME:
            return f"이름이 {MAX_NAME}자를 넘음"
        if name not in self.names:
            self.names.append(name)
            self._save()
        return ""

    def remove(self, name: str) -> None:
        name = clean(name)
        if name in self.names:
            self.names.remove(name)
            self._save()

    def reset(self) -> None:
        self.names = list(self.default)
        if self.path.is_file():
            self.path.unlink()
