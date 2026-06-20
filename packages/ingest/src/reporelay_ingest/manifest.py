from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

ECOSYSTEM_MAP: dict[str, str] = {
    "package.json": "npm",
    "package-lock.json": "npm",
    "requirements.txt": "pypi",
    "setup.py": "pypi",
    "setup.cfg": "pypi",
    "Pipfile": "pypi",
    "Pipfile.lock": "pypi",
    "pyproject.toml": "pypi",
    "Cargo.toml": "cargo",
    "Cargo.lock": "cargo",
    "go.mod": "go",
    "pom.xml": "maven",
    "build.gradle": "gradle",
    "Gemfile": "rubygems",
    "composer.json": "composer",
}

MANIFEST_FILES = list(ECOSYSTEM_MAP.keys())


def detect_manifests(repo_dir: str) -> list[str]:
    """Find dependency manifest files in a repo."""
    import os

    found: list[str] = []
    for fname in MANIFEST_FILES:
        path = os.path.join(repo_dir, fname)
        if os.path.isfile(path):
            found.append(fname)
    return found


def parse_manifest(filename: str, content: str) -> list[dict[str, Any]]:
    """Extract dependencies from a manifest file. Returns list of {name, version, ecosystem, is_dev}."""
    ecosystem = ECOSYSTEM_MAP.get(filename, "unknown")
    deps: list[dict[str, Any]] = []

    try:
        if filename in ("package.json", "package-lock.json"):
            deps.extend(_parse_package_json(content))
        elif filename in ("requirements.txt",):
            deps.extend(_parse_requirements(content))
        elif filename in ("Cargo.toml",):
            deps.extend(_parse_cargo(content))
        elif filename in ("pyproject.toml",):
            deps.extend(_parse_pyproject(content))
    except Exception:
        logger.debug("Failed to parse %s", filename, exc_info=True)

    for dep in deps:
        dep["ecosystem"] = ecosystem
    return deps


def _parse_package_json(content: str) -> list[dict[str, Any]]:
    data = json.loads(content)
    deps: list[dict[str, Any]] = []

    for section, is_dev in [("dependencies", False), ("devDependencies", True), ("peerDependencies", False)]:
        for name, version in data.get(section, {}).items():
            deps.append({"name": name, "version": str(version), "is_dev": is_dev})

    return deps


def _parse_requirements(content: str) -> list[dict[str, Any]]:
    deps: list[dict[str, Any]] = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        match = re.match(r"^([a-zA-Z0-9_.\-]+)\s*([<>=!~].*)?$", line)
        if match:
            deps.append({"name": match.group(1), "version": (match.group(2) or "").strip(), "is_dev": False})
    return deps


def _parse_cargo(content: str) -> list[dict[str, Any]]:
    deps: list[dict[str, Any]] = []
    in_deps = False
    in_dev_deps = False
    for line in content.splitlines():
        line = line.strip()
        if line == "[dependencies]":
            in_deps, in_dev_deps = True, False
            continue
        if line.startswith("[dev-dependencies]"):
            in_deps, in_dev_deps = False, True
            continue
        if line.startswith("[") and (in_deps or in_dev_deps):
            in_deps, in_dev_deps = False, False
            continue
        if (in_deps or in_dev_deps) and "=" in line and not line.startswith("#"):
            name = line.split("=")[0].strip().strip('"')
            version_match = re.search(r'"([^"]*)"', line)
            version = version_match.group(1) if version_match else ""
            deps.append({"name": name, "version": version, "is_dev": in_dev_deps})
    return deps


def _parse_pyproject(content: str) -> list[dict[str, Any]]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib

    data = tomllib.loads(content)
    deps: list[dict[str, Any]] = []

    project = data.get("project", {})
    for dep_str in project.get("dependencies", []):
        name = dep_str.split()[0] if dep_str else ""
        if name:
            deps.append({"name": name, "version": dep_str, "is_dev": False})

    for dep_str in project.get("optional-dependencies", {}).get("dev", []):
        name = dep_str.split()[0] if dep_str else ""
        if name:
            deps.append({"name": name, "version": dep_str, "is_dev": True})

    return deps
