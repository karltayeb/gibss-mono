from __future__ import annotations

import subprocess
import sys

import jax


def _read_x64_flag(module_name: str) -> str:
    command = [
        sys.executable,
        "-c",
        (
            "import jax; "
            f"import {module_name}; "
            "print(jax.config.read('jax_enable_x64'))"
        ),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def test_pytest_starts_with_x64_enabled():
    assert jax.config.read("jax_enable_x64") is True


def test_gibss_import_enables_x64_in_fresh_process():
    assert _read_x64_flag("gibss") == "True"
