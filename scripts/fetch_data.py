#!/usr/bin/env python3
"""Download the pinned MT-Bench question file and verify its SHA256."""
import argparse
import hashlib
import urllib.request
from pathlib import Path

from acceleratellm.data.mt_bench import DEFAULT_PATH, SHA256, URL, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    if args.output.exists():
        verify(args.output)
        print(f"verified {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        payload = response.read()
    if hashlib.sha256(payload).hexdigest() != SHA256:
        raise RuntimeError("Downloaded MT-Bench file does not match the pinned SHA256")
    args.output.write_bytes(payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
