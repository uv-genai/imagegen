"""imagegen — OpenAI-compatible image generation service on stable-diffusion.cpp."""

from importlib.metadata import version

#: Reported by `imagegen --version` and by GET /health. Read from the installed
#: package metadata so pyproject.toml stays the only place a version is written;
#: the package is always installed (editable) in the workflow uv sync documents.
__version__ = version("imagegen")
