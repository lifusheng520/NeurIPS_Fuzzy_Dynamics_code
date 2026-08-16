#!/usr/bin/env python3
"""Check that the offline runtime contains every project dependency."""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


REQUIRED = {
    # Import NumPy before packages that initialize its C API indirectly. Some
    # HPC Python stacks otherwise attempt to initialize the extension twice.
    "numpy": ("numpy", ">=1.26,<3"),
    "torch": ("torch", ">=2.4,<3"),
    "transformers": ("transformers", ">=4.51,<5"),
    "accelerate": ("accelerate", ">=0.27,<2"),
    "datasets": ("datasets", ">=2.17,<5"),
    "pandas": ("pandas", ">=2,<3"),
    "tqdm": ("tqdm", ">=4.66,<5"),
    "requests": ("requests", ">=2.31,<3"),
    "matplotlib": ("matplotlib", ">=3.8,<4"),
    "tuned-lens": ("tuned_lens", "==0.2.0"),
    "Unidecode": ("unidecode", ">=1.3,<2"),
    "fancy-einsum": ("fancy_einsum", "==0.0.3"),
    "PyYAML": ("yaml", ">=6,<7"),
    "sentencepiece": ("sentencepiece", ">=0.2,<1"),
    "safetensors": ("safetensors", ">=0.4,<1"),
    "baukit": ("baukit", ""),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-vllm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"python={platform.python_version()} executable={sys.executable}")
    print(f"platform={platform.platform()}")
    problems: list[str] = []
    if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
        problems.append("Use Python 3.10-3.12 for the full experiment environment.")

    dependencies = dict(REQUIRED)
    if args.require_vllm:
        dependencies["vllm"] = ("vllm", "")
    for distribution, (module, specifier) in dependencies.items():
        try:
            installed_version = version(distribution)
        except PackageNotFoundError:
            installed_version = "not-installed"
        try:
            importlib.import_module(module)
            import_error = None
        except Exception as error:  # dependency/ABI errors must fail the check
            import_error = f"{type(error).__name__}: {error}"
        version_ok = installed_version != "not-installed"
        if version_ok and specifier:
            try:
                version_ok = Version(installed_version) in SpecifierSet(specifier)
            except InvalidVersion:
                version_ok = False
        status = "ok" if import_error is None and version_ok else "invalid"
        print(f"{distribution}={installed_version} [{status}]")
        if import_error is not None:
            problems.append(
                f"Cannot import {distribution} ({module}): {import_error}"
            )
        if not version_ok:
            expected = specifier or "an installed distribution"
            problems.append(
                f"Invalid {distribution} version: {installed_version}; expected {expected}."
            )

    try:
        import torch
    except ImportError:
        pass
    else:
        print(f"torch_cuda_build={torch.version.cuda}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"cuda_device_count={torch.cuda.device_count()}")
        if args.require_cuda and not torch.cuda.is_available():
            problems.append("CUDA was required but torch.cuda.is_available() is false.")

    if problems:
        print("\nEnvironment check failed:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        raise SystemExit(1)
    print("Environment check passed.")


if __name__ == "__main__":
    main()
