# SPDX-License-Identifier: LicenseRef-SnapKitty-Commercial-1.0
# Copyright (C) 2026 SnapKitty Collective
"""Unit tests for scripts/verify.py.

Builds a minimal synthetic GGUF header in a temp dir (never the real
2.5 GB file) and asserts the verifier accepts it, plus negative cases.
"""

import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import verify  # noqa: E402


def _wstr(buf: bytearray, s: str) -> None:
    b = s.encode("utf-8")
    buf += struct.pack("<Q", len(b)) + b


def _wkv_str(buf: bytearray, key: str, val: str) -> None:
    _wstr(buf, key)
    buf += struct.pack("<I", 8)  # string type
    _wstr(buf, val)


def _wkv_u32(buf: bytearray, key: str, val: int) -> None:
    _wstr(buf, key)
    buf += struct.pack("<I", 4)  # u32 type
    buf += struct.pack("<I", val)


def make_gguf(path: str, arch="nemotron", file_type=15,
              with_license=True) -> None:
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)   # version
    buf += struct.pack("<Q", 0)   # tensor count
    kvs = bytearray()
    _wkv_str(kvs, "general.architecture", arch)
    _wkv_str(kvs, "general.name", "Minitron 4B Instruct")
    _wkv_u32(kvs, "general.file_type", file_type)
    _wkv_str(kvs, "general.base_model.0.repo_url",
             "https://huggingface.co/nvidia/Minitron-4B-Base")
    if with_license:
        _wkv_str(kvs, "general.license.name",
                 "nvidia-community-model-license")
        _wkv_str(kvs, "general.license.link",
                 "https://huggingface.co/nvidia/Nemotron-Mini-4B-Instruct")
    buf += struct.pack("<Q", 6 if with_license else 4)
    buf += kvs
    with open(path, "wb") as fh:
        fh.write(bytes(buf))


def _passed(results):
    return all(p for _, p, _ in results)


def test_accepts_valid_header(tmp_path):
    p = str(tmp_path / "good.gguf")
    make_gguf(p)
    results = verify.check(p)
    assert _passed(results), results
    assert len(results) == 5


def test_rejects_wrong_architecture(tmp_path):
    p = str(tmp_path / "bad.gguf")
    make_gguf(p, arch="llama")
    results = verify.check(p)
    assert not _passed(results)
    assert [n for n, ok, _ in results if n == "architecture"][0]


def test_rejects_wrong_file_type(tmp_path):
    p = str(tmp_path / "bad.gguf")
    make_gguf(p, file_type=2)  # not Q4_K_M
    results = verify.check(p)
    assert not _passed(results)


def test_rejects_missing_nvidia_license(tmp_path):
    p = str(tmp_path / "bad.gguf")
    make_gguf(p, with_license=False)
    results = verify.check(p)
    assert not _passed(results)


def test_rejects_non_gguf(tmp_path):
    p = str(tmp_path / "nope.bin")
    with open(p, "wb") as fh:
        fh.write(b"definitely not a gguf file")
    results = verify.check(p)
    assert not _passed(results)
    assert results[0][0] == "parse"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
