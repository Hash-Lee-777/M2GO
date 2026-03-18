from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""


setup(
    name="M2GO",
    version="0.0.0",
    description="M2GO: multi-modal / 2D-3D domain adaptation toolkit",
    long_description=README,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("configs", "data", "traintxt", "output", "analysis")),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[],
)

