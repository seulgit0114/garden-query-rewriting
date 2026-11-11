#!/usr/bin/env python3
"""Run provenance, atomic writes, append-only log, checkpoint/resume.

Everything that makes a long collection restartable and auditable lives here so
the search controller stays about search.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import typing as t
from datetime import datetime, timezone
from pathlib import Path

from . import config


def git_head(repo: Path) -> t.Dict[str, str]:
    def run(*args):
        try:
            return subprocess.run(['git', '-C', str(repo), *args],
                                  capture_output=True, text=True, timeout=15).stdout.strip()
        except Exception:                                          # noqa: BLE001
            return ''
    return {'repo': str(repo), 'head': run('rev-parse', 'HEAD'),
            'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty_count': str(len([l for l in run('status', '--porcelain').splitlines() if l]))}


def atomic_write_json(path: Path, obj: t.Any) -> None:
    """Write via temp file + os.replace so a crash never leaves a half file.

    The temp file is created in the destination directory: os.replace is only
    atomic within a filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class TransitionLog:
    """Append-only JSONL sink. One flush per row so a kill loses at most one."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self.n_written = 0

    def __enter__(self):
        self._fh = self.path.open('a')
        return self

    def __exit__(self, *exc):
        if self._fh:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None
        return False

    def write(self, row: dict) -> None:
        self._fh.write(json.dumps(row, ensure_ascii=False) + '\n')
        self._fh.flush()
        self.n_written += 1


class Checkpoint:
    """Query-level resume.

    Granularity is deliberate: a query is the unit that can be re-run cheaply
    and is also the unit of the fold assignment, so resuming mid-query would
    risk mixing two runs' states inside one query's subtree.
    """

    def __init__(self, path: Path):
        self.path = path
        self.state = {'completed': [], 'failed': {}, 'started_utc': None, 'updated_utc': None}
        if path.exists():
            try:
                self.state.update(json.loads(path.read_text()))
            except Exception:                                      # noqa: BLE001
                pass

    @property
    def completed(self) -> set:
        return set(tuple(x) if isinstance(x, list) else x for x in self.state['completed'])

    def is_done(self, key: t.Tuple[str, str]) -> bool:
        return list(key) in self.state['completed'] or key in self.completed

    def mark_done(self, key: t.Tuple[str, str]) -> None:
        if not self.is_done(key):
            self.state['completed'].append(list(key))
        self._save()

    def mark_failed(self, key: t.Tuple[str, str], reason: str) -> None:
        self.state['failed'][f'{key[0]}/{key[1]}'] = reason
        self._save()

    def _save(self) -> None:
        self.state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        if self.state['started_utc'] is None:
            self.state['started_utc'] = self.state['updated_utc']
        atomic_write_json(self.path, self.state)


def build_manifest(kind: str, args: dict, inputs: dict, seed: int) -> dict:
    return {
        'kind': kind,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'seed': seed,
        'python': sys.version.split()[0],
        'argv': sys.argv,
        'args': args,
        'inputs': inputs,
        'git': {
            'llm4rewrite': git_head(config.PROJECT_ROOT),
            'rbot_baseline': git_head(config.BASELINE_WORKTREE),
        },
        'limits': {
            'max_depth': config.DEFAULT_MAX_DEPTH,
            'max_states_per_query': config.DEFAULT_MAX_STATES_PER_QUERY,
            'max_calls_per_query': config.DEFAULT_MAX_CALLS_PER_QUERY,
            'max_rules_per_state': config.DEFAULT_MAX_RULES_PER_STATE,
        },
    }
