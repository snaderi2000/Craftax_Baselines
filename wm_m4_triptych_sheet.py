#!/usr/bin/env python3
"""
M4 world-model triptych wrapper.

Uses the shared triptych implementation from wm_m3_triptych_sheet.py, but
applies M4-friendly defaults:
  - tokenizer_type=nnt
  - vqvae_codebook_size=4096
  - auto_vocab_from_checkpoint=true
"""

import sys

from wm_m3_triptych_sheet import main as _shared_main


def _has_flag(argv, flag_name: str) -> bool:
    return any(a == flag_name or a.startswith(flag_name + "=") for a in argv)


def _inject_m4_defaults(argv):
    out = list(argv)

    if not _has_flag(out, "--tokenizer_type"):
        out += ["--tokenizer_type", "nnt"]

    # Keep M4 default vocab, but still allow auto override from checkpoint.
    if not _has_flag(out, "--vqvae_codebook_size"):
        out += ["--vqvae_codebook_size", "4096"]

    if not _has_flag(out, "--auto_vocab_from_checkpoint") and not _has_flag(out, "--no-auto_vocab_from_checkpoint"):
        out += ["--auto_vocab_from_checkpoint"]

    return out


if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + _inject_m4_defaults(sys.argv[1:])
    _shared_main()

