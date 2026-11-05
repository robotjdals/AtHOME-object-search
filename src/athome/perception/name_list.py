"""Editable list of detector names, kept in a JSON file: the fixed list changed
from the perception check's page. With CG-D the search reads the same file, at
every image (read_names), so a change applies to it at once; the default list
(configs/perception) stays as it is and returns when the file is removed."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Sequence

MAX_NAME = 60


def clean(name: str) -> str:
    return " ".join(str(name).casefold().split())


def read_names(path: Path, default: Sequence[str]) -> List[str]:
    """The saved names, the default list without a file."""
    path = Path(path)
    if not path.is_file():
        return list(default)
    saved = json.loads(path.read_text(encoding="utf-8"))
    return list(dict.fromkeys(clean(n) for n in saved["names"] if clean(n)))


class NameList:
    def __init__(self, path: Path, default: Sequence[str]):
        self.path = Path(path)
        self.default = list(dict.fromkeys(clean(n) for n in default))
        self.names: List[str] = read_names(self.path, self.default)

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

    def rename(self, old: str, new: str) -> str:
        """The error for the page, '' when renamed; in the same place of the list."""
        old, new = clean(old), clean(new)
        if old not in self.names:
            return "목록에 없는 이름"
        if not new:
            return "이름이 비어 있음"
        if len(new) > MAX_NAME:
            return f"이름이 {MAX_NAME}자를 넘음"
        if new != old:
            if new in self.names:
                return "이미 있는 이름"
            self.names[self.names.index(old)] = new
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
