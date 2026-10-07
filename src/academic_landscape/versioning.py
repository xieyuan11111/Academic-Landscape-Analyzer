"""FR-11 analysis versioning.

Every analysis result is bound to a corpus snapshot, configuration, model and
generation timestamp. Subsequent analyses — after adding/removing/excluding
documents or applying human corrections — produce a new version. Old versions
are never silently overwritten.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LIST_KEY_TO_DIFF_KEYS: dict[str, tuple[str, str]] = {
    "papers": ("added_papers", "removed_papers"),
    "directions": ("added_directions", "removed_directions"),
    "relations": ("added_relations", "removed_relations"),
    "frontier_candidates": ("added_frontiers", "removed_frontiers"),
}


def _canonical_json_dumps(value: Any) -> str:
    """Serialize *value* to a canonical JSON string (sorted keys, no whitespace)."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonicalize(value: Any) -> Any:
    """Recursively normalize a value so dict key order no longer matters."""
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value.keys())}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value


def _safe_list(payload: dict | None, key: str) -> list[Any]:
    """Return payload[key] if it is a list, else an empty list (None-safe)."""
    if not isinstance(payload, dict):
        return []
    value = payload.get(key)
    if isinstance(value, list):
        return value
    return []


@dataclass(frozen=True)
class AnalysisVersion:
    version_id: str  # 'ver_' + sha256(snapshot_id|config_hash|model|prompt_version)[:16]
    snapshot_id: str
    config_hash: str  # output of compute_config_hash(config)
    model: str
    prompt_version: str
    created_at: str  # ISO8601 UTC
    parent_version_id: str = ""  # derived versions point at their parent
    note: str = ""  # e.g. 'human correction' / 'added 12 papers'


def compute_config_hash(config: dict) -> str:
    """sha256 of a canonical (sorted-key) JSON encoding of *config*."""
    canonical = _canonicalize(config or {})
    payload = _canonical_json_dumps(canonical)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def new_version(
    *,
    snapshot_id: str,
    config: dict,
    model: str,
    prompt_version: str,
    parent: AnalysisVersion | None = None,
    note: str = "",
) -> AnalysisVersion:
    """Build a fresh AnalysisVersion for the given inputs."""
    config_hash = compute_config_hash(config)
    raw = f"{snapshot_id}|{config_hash}|{model}|{prompt_version}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    version_id = f"ver_{digest}"
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return AnalysisVersion(
        version_id=version_id,
        snapshot_id=snapshot_id,
        config_hash=config_hash,
        model=model,
        prompt_version=prompt_version,
        created_at=created_at,
        parent_version_id=parent.version_id if parent is not None else "",
        note=note,
    )


def diff_versions(base: dict, target: dict) -> dict:
    """Diff two analysis payloads. Missing keys / non-list values are treated as empty."""
    diff: dict[str, list[Any]] = {}
    for list_key, (added_key, removed_key) in _LIST_KEY_TO_DIFF_KEYS.items():
        base_list = _safe_list(base, list_key)
        target_list = _safe_list(target, list_key)
        base_canonical = {_canonical_json_dumps(item) for item in base_list}
        # Preserve input order for both added (target order) and removed (base order)
        diff[added_key] = [
            item for item in target_list
            if _canonical_json_dumps(item) not in base_canonical
        ]
        diff[removed_key] = [
            item for item in base_list
            if _canonical_json_dumps(item) not in {_canonical_json_dumps(i) for i in target_list}
        ]
    return diff


class VersionStore:
    """Filesystem-backed store for analysis versions and their payloads."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, version_id: str) -> Path:
        return self.root / f"{version_id}.json"

    @staticmethod
    def _build_record(version: AnalysisVersion, payload: dict) -> dict:
        return {"version": asdict(version), "payload": payload}

    @staticmethod
    def _record_to_version(record: dict) -> AnalysisVersion:
        v = record["version"]
        return AnalysisVersion(
            version_id=v["version_id"],
            snapshot_id=v["snapshot_id"],
            config_hash=v["config_hash"],
            model=v["model"],
            prompt_version=v["prompt_version"],
            created_at=v["created_at"],
            parent_version_id=v.get("parent_version_id", ""),
            note=v.get("note", ""),
        )

    def save(self, version: AnalysisVersion, payload: dict) -> Path:
        """Persist *version + payload*. Idempotent for identical content;
        raises ValueError if an existing record with the same id has different content."""
        path = self._path_for(version.version_id)
        record = self._build_record(version, payload)
        new_content = _canonical_json_dumps(record)
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            if existing != new_content:
                raise ValueError(
                    f"version {version.version_id} already exists with different content"
                )
            return path
        path.write_text(new_content, encoding="utf-8")
        return path

    def load(self, version_id: str) -> tuple[AnalysisVersion, dict]:
        path = self._path_for(version_id)
        if not path.exists():
            raise KeyError(f"unknown version_id: {version_id}")
        record = json.loads(path.read_text(encoding="utf-8"))
        return self._record_to_version(record), record.get("payload", {})

    def list(self) -> list[AnalysisVersion]:
        """All versions in the store, sorted by created_at ascending."""
        versions: list[AnalysisVersion] = []
        for path in sorted(self.root.glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            versions.append(self._record_to_version(record))
        versions.sort(key=lambda v: v.created_at)
        return versions