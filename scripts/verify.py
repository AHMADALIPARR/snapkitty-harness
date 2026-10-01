# SPDX-License-Identifier: LicenseRef-SnapKitty-Commercial-1.0
# Copyright (C) 2026 SnapKitty Collective
"""Verify a snapkitty-harness GGUF artifact from its KV header.

Pure stdlib (struct) — no torch, no numpy, no network. Parses the GGUF
metadata the same way the original artifact's header was parsed and
asserts the expected provenance fields:

  - magic == b'GGUF'
  - general.architecture == 'nemotron'
  - general.file_type == 15            (Q4_K_M)
  - general.name mentions Minitron 4B Instruct
  - general.base_model.0.repo_url -> nvidia/Minitron-4B-Base
  - NVIDIA license fields present (general.license.name / .link)

Exit 0 when every check passes, 1 otherwise.

Usage:
    python verify.py path/to/snapkitty-harness.Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import struct
import sys

# GGUF value types
_T_U8, _T_I8, _T_U16, _T_I16 = 0, 1, 2, 3
_T_U32, _T_I32, _T_F32, _T_BOOL = 4, 5, 6, 7
_T_STR, _T_ARR = 8, 9
_T_U64, _T_I64, _T_F64 = 10, 11, 12

_FIXED = {
    _T_U8: 1, _T_I8: 1, _T_U16: 2, _T_I16: 2,
    _T_U32: 4, _T_I32: 4, _T_F32: 4, _T_BOOL: 1,
    _T_U64: 8, _T_I64: 8, _T_F64: 8,
}


def _read_str(data: bytes, off: int) -> tuple[str, int]:
    (length,) = struct.unpack_from("<Q", data, off)
    off += 8
    s = data[off:off + length].decode("utf-8", "replace")
    return s, off + length


def _skip_value(data: bytes, off: int, vtype: int) -> int:
    if vtype in _FIXED:
        return off + _FIXED[vtype]
    if vtype == _T_STR:
        _, off = _read_str(data, off)
        return off
    if vtype == _T_ARR:
        (elem_type,) = struct.unpack_from("<I", data, off)
        off += 4
        (count,) = struct.unpack_from("<Q", data, off)
        off += 8
        for _ in range(count):
            off = _skip_value(data, off, elem_type)
        return off
    raise ValueError(f"unknown GGUF value type {vtype}")


def _get_value(data: bytes, off: int, vtype: int):
    if vtype in _FIXED:
        fmt = {_T_U8: "<B", _T_I8: "<b", _T_U16: "<H", _T_I16: "<h",
               _T_U32: "<I", _T_I32: "<i", _T_F32: "<f", _T_BOOL: "<?",
               _T_U64: "<Q", _T_I64: "<q", _T_F64: "<d"}[vtype]
        (val,) = struct.unpack_from(fmt, data, off)
        return val
    if vtype == _T_STR:
        s, _ = _read_str(data, off)
        return s
    if vtype == _T_ARR:
        (elem_type,) = struct.unpack_from("<I", data, off)
        off += 4
        (count,) = struct.unpack_from("<Q", data, off)
        off += 8
        vals = []
        for _ in range(count):
            v = _get_value(data, off, elem_type)
            vals.append(v)
            off = _skip_value(data, off, elem_type)
        return vals
    raise ValueError(f"unknown GGUF value type {vtype}")


def parse_header(path: str, max_kv_bytes: int = 64 * 1024 * 1024) -> dict:
    """Parse the GGUF header KV section into a flat dict."""
    with open(path, "rb") as fh:
        head = fh.read(max_kv_bytes)
    if head[:4] != b"GGUF":
        raise ValueError("not a GGUF file (bad magic)")
    off = 4
    (version,) = struct.unpack_from("<I", head, off)
    off += 4
    (n_tensors,) = struct.unpack_from("<Q", head, off)
    off += 8
    (n_kv,) = struct.unpack_from("<Q", head, off)
    off += 8
    kv: dict = {}
    for _ in range(n_kv):
        key, off = _read_str(head, off)
        (vtype,) = struct.unpack_from("<I", head, off)
        off += 4
        kv[key] = _get_value(head, off, vtype)
        off = _skip_value(head, off, vtype)
    return {"version": version, "n_tensors": n_tensors, "kv": kv}


def check(path: str) -> list[tuple[str, bool, str]]:
    """Run provenance assertions. Returns (name, passed, detail)."""
    results: list[tuple[str, bool, str]] = []
    try:
        header = parse_header(path)
    except (OSError, ValueError, struct.error) as exc:
        return [("parse", False, f"could not parse GGUF header: {exc}")]
    kv = header["kv"]

    def get(key: str):
        return kv.get(key)

    arch = get("general.architecture")
    results.append(("architecture",
                    arch == "nemotron",
                    f"general.architecture={arch!r} (want 'nemotron')"))

    ftype = get("general.file_type")
    results.append(("file_type",
                    ftype == 15,
                    f"general.file_type={ftype!r} (want 15 = Q4_K_M)"))

    name = get("general.name") or ""
    results.append(("model_name",
                    "minitron" in name.lower() and "4b" in name.lower(),
                    f"general.name={name!r}"))

    base_url = get("general.base_model.0.repo_url") or ""
    results.append(("base_model",
                    "nvidia/minitron-4b-base" in base_url.lower(),
                    f"general.base_model.0.repo_url={base_url!r}"))

    lic_name = get("general.license.name") or ""
    lic_link = get("general.license.link") or ""
    results.append(("nvidia_license",
                    bool(lic_name) and bool(lic_link),
                    f"general.license.name={lic_name!r} "
                    f"general.license.link={lic_link!r}"))

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("gguf", help="path to the GGUF file to verify")
    args = ap.parse_args()

    results = check(args.gguf)
    ok = True
    for name, passed, detail in results:
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {name}: {detail}")
        ok = ok and passed
    print("verify: OK" if ok else "verify: FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
