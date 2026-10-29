#!/usr/bin/env python3
"""Evaluate version-specific dependency licenses against deployment-context policy.

The inventory is produced by OSV-Scanner from committed lockfiles. This module
adds Conductor's deployment-context policy, runtime/dev scope handling,
exceptions, vendored-source checks, and a committed violation baseline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


STATUS_ORDER = {"allow": 0, "info": 0, "warn": 1, "deny": 2}
SOURCE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".m",
    ".mm",
    ".py",
    ".rb",
    ".rs",
    ".swift",
    ".ts",
    ".tsx",
}
VENDOR_DIR_NAMES = {
    "deps",
    "extern",
    "external",
    "third-party",
    "third_party",
    "vendor",
    "vendored",
}
LICENSE_FILENAMES = {
    "copying",
    "copying.md",
    "copying.txt",
    "license",
    "license.md",
    "license.txt",
    "licenses",
    "notice",
    "notice.md",
    "notice.txt",
}


def normalize_name(name: str, ecosystem: str = "") -> str:
    name = name.strip()
    if ecosystem.lower() in {"pypi", "python"}:
        return re.sub(r"[-_.]+", "-", name).lower()
    return name.lower()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PolicyError(f"required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PolicyError(f"invalid JSON in {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class PolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PackageKey:
    ecosystem: str
    name: str
    version: str


@dataclass
class Finding:
    code: str
    status: str
    context: str
    scope: str
    ecosystem: str
    package: str
    version: str
    license_expression: str
    source: str
    detail: str
    fingerprint: str = ""

    def finalize(self) -> "Finding":
        material = "|".join(
            [
                self.code,
                self.context,
                self.scope,
                self.ecosystem,
                self.package,
                self.version,
                self.license_expression,
                self.source,
            ]
        )
        self.fingerprint = hashlib.sha256(material.encode()).hexdigest()[:20]
        return self

    def as_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "status": self.status,
            "context": self.context,
            "scope": self.scope,
            "ecosystem": self.ecosystem,
            "package": self.package,
            "version": self.version,
            "license": self.license_expression,
            "source": self.source,
            "detail": self.detail,
            "fingerprint": self.fingerprint,
        }


def clean_source_path(raw: str, repo_root: Path) -> str:
    raw = raw.replace("\\", "/")
    roots = [
        str(repo_root.resolve()).replace("\\", "/"),
        "/github/workspace",
        "/workspace",
        "/src",
    ]
    for root in roots:
        prefix = root.rstrip("/") + "/"
        if raw.startswith(prefix):
            return raw[len(prefix) :]
    return raw.lstrip("./")


def package_name_from_requirement(value: str) -> str:
    value = value.strip()
    if value.startswith(("-e ", "--editable ")):
        return ""
    match = re.match(r"([A-Za-z0-9_.-]+(?:\[[^]]+\])?)", value)
    if not match:
        return ""
    return match.group(1).split("[", 1)[0]


def collect_first_party_names(repo_root: Path) -> set[str]:
    names: set[str] = set()
    ignored = {".git", ".venv", "node_modules", "vendor", "vendored"}
    for path in repo_root.rglob("pyproject.toml"):
        if any(part in ignored for part in path.parts):
            continue
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        for value in (
            data.get("project", {}).get("name"),
            data.get("tool", {}).get("poetry", {}).get("name"),
        ):
            if isinstance(value, str):
                names.add(normalize_name(value, "PyPI"))
    for path in repo_root.rglob("package.json"):
        if any(part in ignored for part in path.parts):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8")).get("name")
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, str):
            names.add(normalize_name(value, "npm"))
    for path in repo_root.rglob("setup.py"):
        if any(part in ignored for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = re.search(r"\bname\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            names.add(normalize_name(match.group(1), "PyPI"))
    return names


class ScopeIndex:
    """Classify packages in a lockfile as runtime or dev-only."""

    def __init__(self, repo_root: Path, dev_groups: set[str]) -> None:
        self.repo_root = repo_root
        self.dev_groups = {group.lower() for group in dev_groups}
        self.cache: dict[str, dict[PackageKey, str]] = {}

    def scope(self, source: str, key: PackageKey) -> str:
        if source not in self.cache:
            self.cache[source] = self._build(source)
        index = self.cache[source]
        exact = index.get(key)
        if exact:
            return exact
        same_name = [
            scope
            for candidate, scope in index.items()
            if candidate.ecosystem == key.ecosystem and candidate.name == key.name
        ]
        if "runtime" in same_name:
            return "runtime"
        if "dev" in same_name:
            return "dev"
        filename = Path(source).name.lower()
        if any(word in filename for word in ("dev", "test", "lint", "docs")):
            return "dev"
        # Fail closed when the lock format cannot establish dev-only status.
        return "runtime"

    def _build(self, source: str) -> dict[PackageKey, str]:
        path = self.repo_root / source
        if not path.is_file():
            return {}
        name = path.name.lower()
        try:
            if name == "package-lock.json":
                return self._package_lock(path)
            if name == "pnpm-lock.yaml":
                return self._pnpm_lock(path)
            if name == "uv.lock":
                return self._uv_lock(path)
            if name == "poetry.lock":
                return self._poetry_lock(path)
        except (OSError, ValueError, KeyError, json.JSONDecodeError, tomllib.TOMLDecodeError):
            return {}
        return {}

    @staticmethod
    def _merge(index: dict[PackageKey, str], key: PackageKey, scope: str) -> None:
        if index.get(key) == "runtime":
            return
        index[key] = scope

    def _package_lock(self, path: Path) -> dict[PackageKey, str]:
        data = json.loads(path.read_text(encoding="utf-8"))
        index: dict[PackageKey, str] = {}
        for location, meta in data.get("packages", {}).items():
            if not location or not isinstance(meta, dict):
                continue
            package_name = meta.get("name") or self._npm_name_from_location(location)
            version = str(meta.get("version", ""))
            if not package_name or not version:
                continue
            scope = "dev" if meta.get("dev") and not meta.get("optional") else "runtime"
            key = PackageKey("npm", normalize_name(package_name, "npm"), version)
            self._merge(index, key, scope)
        return index

    @staticmethod
    def _npm_name_from_location(location: str) -> str:
        parts = location.replace("\\", "/").split("node_modules/")[-1].split("/")
        if parts and parts[0].startswith("@") and len(parts) > 1:
            return "/".join(parts[:2])
        return parts[0] if parts else ""

    def _uv_lock(self, path: Path) -> dict[PackageKey, str]:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        packages = [item for item in data.get("package", []) if isinstance(item, dict)]
        by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for package in packages:
            by_name[normalize_name(str(package.get("name", "")), "PyPI")].append(package)

        roots = []
        for package in packages:
            source = package.get("source", {})
            if isinstance(source, dict) and set(source).intersection(
                {"directory", "editable", "virtual", "workspace"}
            ):
                roots.append(package)
        if not roots:
            first_party = collect_first_party_names(path.parent)
            roots = [
                package
                for package in packages
                if normalize_name(str(package.get("name", "")), "PyPI") in first_party
            ]

        runtime_seeds: set[str] = set()
        dev_seeds: set[str] = set()
        for root in roots:
            runtime_seeds.update(self._dependency_names(root.get("dependencies", []), "PyPI"))
            optional = root.get("optional-dependencies", {})
            if isinstance(optional, dict):
                for dependencies in optional.values():
                    runtime_seeds.update(self._dependency_names(dependencies, "PyPI"))
            dev = root.get("dev-dependencies", {})
            if isinstance(dev, dict):
                for dependencies in dev.values():
                    dev_seeds.update(self._dependency_names(dependencies, "PyPI"))

        runtime = self._walk_python_graph(runtime_seeds, by_name)
        dev = self._walk_python_graph(dev_seeds, by_name) - runtime
        index: dict[PackageKey, str] = {}
        for package in packages:
            name = normalize_name(str(package.get("name", "")), "PyPI")
            version = str(package.get("version", ""))
            if not name or not version:
                continue
            key = PackageKey("PyPI", name, version)
            if name in runtime:
                index[key] = "runtime"
            elif name in dev:
                index[key] = "dev"
        return index

    def _poetry_lock(self, path: Path) -> dict[PackageKey, str]:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        packages = [item for item in data.get("package", []) if isinstance(item, dict)]
        by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for package in packages:
            by_name[normalize_name(str(package.get("name", "")), "PyPI")].append(package)

        pyproject_path = path.with_name("pyproject.toml")
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        runtime_seeds: set[str] = set()
        dev_seeds: set[str] = set()
        project = pyproject.get("project", {})
        runtime_seeds.update(
            normalize_name(package_name_from_requirement(item), "PyPI")
            for item in project.get("dependencies", [])
            if package_name_from_requirement(item)
        )
        for group, dependencies in project.get("optional-dependencies", {}).items():
            target = dev_seeds if group.lower() in self.dev_groups else runtime_seeds
            target.update(
                normalize_name(package_name_from_requirement(item), "PyPI")
                for item in dependencies
                if package_name_from_requirement(item)
            )

        poetry = pyproject.get("tool", {}).get("poetry", {})
        for dep in poetry.get("dependencies", {}):
            if dep.lower() != "python":
                runtime_seeds.add(normalize_name(dep, "PyPI"))
        for dep in poetry.get("dev-dependencies", {}):
            dev_seeds.add(normalize_name(dep, "PyPI"))
        for group, value in poetry.get("group", {}).items():
            target = dev_seeds if group.lower() in self.dev_groups else runtime_seeds
            target.update(normalize_name(dep, "PyPI") for dep in value.get("dependencies", {}))

        runtime = self._walk_python_graph(runtime_seeds, by_name)
        dev = self._walk_python_graph(dev_seeds, by_name) - runtime
        index: dict[PackageKey, str] = {}
        for package in packages:
            name = normalize_name(str(package.get("name", "")), "PyPI")
            version = str(package.get("version", ""))
            key = PackageKey("PyPI", name, version)
            if name in runtime:
                index[key] = "runtime"
            elif name in dev:
                index[key] = "dev"
        return index

    @staticmethod
    def _dependency_names(dependencies: Any, ecosystem: str) -> set[str]:
        names: set[str] = set()
        if isinstance(dependencies, dict):
            values: Iterable[Any] = dependencies.keys()
        elif isinstance(dependencies, list):
            values = dependencies
        else:
            return names
        for value in values:
            if isinstance(value, dict):
                value = value.get("name", "")
            if isinstance(value, str) and value:
                names.add(normalize_name(value, ecosystem))
        return names

    def _walk_python_graph(
        self, seeds: set[str], by_name: dict[str, list[dict[str, Any]]]
    ) -> set[str]:
        visited: set[str] = set()
        queue = deque(seeds)
        while queue:
            name = queue.popleft()
            if not name or name in visited:
                continue
            visited.add(name)
            for package in by_name.get(name, []):
                queue.extend(self._dependency_names(package.get("dependencies", []), "PyPI"))
        return visited

    def _pnpm_lock(self, path: Path) -> dict[PackageKey, str]:
        lines = path.read_text(encoding="utf-8").splitlines()
        direct_runtime: list[tuple[str, str]] = []
        direct_dev: list[tuple[str, str]] = []
        nodes: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
        top = ""
        importer_section = ""
        direct_name = ""
        snapshot: tuple[str, str] | None = None
        snapshot_section = ""

        for raw in lines:
            stripped = raw.lstrip()
            if not stripped or stripped.startswith("#"):
                continue
            indent = len(raw) - len(stripped)
            if indent == 0 and stripped.endswith(":"):
                top = stripped[:-1]
                importer_section = ""
                snapshot = None
                continue
            if top == "importers":
                if indent == 4 and stripped.endswith(":"):
                    importer_section = stripped[:-1]
                    direct_name = ""
                elif indent == 6 and stripped.endswith(":") and importer_section in {
                    "dependencies",
                    "optionalDependencies",
                    "devDependencies",
                }:
                    direct_name = self._yaml_key(stripped[:-1])
                elif indent == 8 and stripped.startswith("version:") and direct_name:
                    version = self._pnpm_ref(stripped.split(":", 1)[1])
                    target = direct_dev if importer_section == "devDependencies" else direct_runtime
                    target.append((normalize_name(direct_name, "npm"), version))
            elif top == "snapshots":
                if indent == 2 and stripped.endswith(":"):
                    snapshot = self._pnpm_snapshot_key(self._yaml_key(stripped[:-1]))
                    snapshot_section = ""
                elif indent == 4 and stripped.endswith(":"):
                    snapshot_section = stripped[:-1]
                elif (
                    indent == 6
                    and snapshot
                    and snapshot_section in {"dependencies", "optionalDependencies"}
                    and ":" in stripped
                ):
                    dep_name, dep_ref = stripped.split(":", 1)
                    nodes[snapshot].append(
                        (normalize_name(self._yaml_key(dep_name), "npm"), self._pnpm_ref(dep_ref))
                    )

        runtime = self._walk_pnpm_graph(direct_runtime, nodes)
        dev = self._walk_pnpm_graph(direct_dev, nodes) - runtime
        index: dict[PackageKey, str] = {}
        for name, version in nodes:
            key = PackageKey("npm", name, version)
            if (name, version) in runtime:
                index[key] = "runtime"
            elif (name, version) in dev:
                index[key] = "dev"
        return index

    @staticmethod
    def _yaml_key(value: str) -> str:
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]
        return value

    @staticmethod
    def _pnpm_ref(value: str) -> str:
        value = value.strip().strip("'\"")
        if value.startswith("npm:"):
            value = value.rsplit("@", 1)[-1]
        return value.split("(", 1)[0]

    @classmethod
    def _pnpm_snapshot_key(cls, value: str) -> tuple[str, str] | None:
        value = value.split("(", 1)[0]
        if "@" not in value:
            return None
        name, version = value.rsplit("@", 1)
        if not name or not version:
            return None
        return normalize_name(name, "npm"), cls._pnpm_ref(version)

    @staticmethod
    def _walk_pnpm_graph(
        seeds: list[tuple[str, str]],
        nodes: dict[tuple[str, str], list[tuple[str, str]]],
    ) -> set[tuple[str, str]]:
        by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for node in nodes:
            by_name[node[0]].append(node)
        visited: set[tuple[str, str]] = set()
        queue: deque[tuple[str, str]] = deque()
        for name, version in seeds:
            exact = (name, version)
            if exact in nodes:
                queue.append(exact)
            else:
                queue.extend(by_name.get(name, []))
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for dep_name, dep_version in nodes.get(node, []):
                exact = (dep_name, dep_version)
                if exact in nodes:
                    queue.append(exact)
                else:
                    queue.extend(by_name.get(dep_name, []))
        return visited


class ExpressionParser:
    def __init__(self, expression: str) -> None:
        self.tokens = re.findall(r"\(|\)|\bAND\b|\bOR\b|\bWITH\b|[^\s()]+", expression)
        self.position = 0

    def parse(self) -> Any:
        if not self.tokens:
            return ("leaf", "UNKNOWN")
        node = self._or()
        if self.position != len(self.tokens):
            raise ValueError("unparsed SPDX tokens")
        return node

    def _or(self) -> Any:
        node = self._and()
        while self._peek() == "OR":
            self.position += 1
            node = ("or", node, self._and())
        return node

    def _and(self) -> Any:
        node = self._primary()
        while self._peek() == "AND":
            self.position += 1
            node = ("and", node, self._primary())
        return node

    def _primary(self) -> Any:
        if self._peek() == "(":
            self.position += 1
            node = self._or()
            if self._peek() != ")":
                raise ValueError("missing closing parenthesis")
            self.position += 1
            return node
        token = self._peek()
        if token is None or token in {"AND", "OR", "WITH", ")"}:
            raise ValueError("expected SPDX license identifier")
        self.position += 1
        if self._peek() == "WITH":
            self.position += 1
            exception = self._peek()
            if exception is None:
                raise ValueError("missing SPDX exception identifier")
            self.position += 1
            return ("leaf", token, exception)
        return ("leaf", token)

    def _peek(self) -> str | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None


def leaves(node: Any) -> set[str]:
    if node[0] == "leaf":
        return {node[1]}
    return leaves(node[1]) | leaves(node[2])


class Policy:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.rules = config["license_rules"]
        self.unknown_policy = config.get("rollout", {}).get("unknown", "deny")

    def context_for(self, source: str) -> str:
        for item in self.config.get("path_contexts", []):
            if fnmatch.fnmatch(source, item["pattern"]):
                return item["context"]
        return self.config["default_context"]

    def evaluate_expression(
        self, expression: str, context: str, elected_license: str = ""
    ) -> tuple[str, str, str]:
        try:
            tree = ExpressionParser(expression).parse()
        except ValueError:
            status = "warn" if self.unknown_policy == "warn" else "deny"
            return status, "UNPARSEABLE_LICENSE", "unparseable license expression"
        return self._evaluate_node(tree, context, elected_license)

    def _evaluate_node(
        self, node: Any, context: str, elected_license: str
    ) -> tuple[str, str, str]:
        operator = node[0]
        if operator == "leaf":
            return self._evaluate_leaf(node[1], context)
        if operator == "and":
            left = self._evaluate_node(node[1], context, elected_license)
            right = self._evaluate_node(node[2], context, elected_license)
            worst = max((left, right), key=lambda item: STATUS_ORDER[item[0]])
            return worst[0], "AND_" + worst[1], "all AND terms apply; " + worst[2]
        if operator == "or":
            options = leaves(node)
            if elected_license and elected_license in options:
                selected = self._evaluate_leaf(elected_license, context)
                return selected[0], "ELECTED_OR", f"elected {elected_license}; {selected[2]}"
            left = self._evaluate_node(node[1], context, "")
            right = self._evaluate_node(node[2], context, "")
            worst = max((left, right), key=lambda item: STATUS_ORDER[item[0]])
            status = worst[0] if STATUS_ORDER[worst[0]] >= STATUS_ORDER["warn"] else "warn"
            return status, "UNELECTED_OR", "no OR option is elected; most restrictive option applies"
        raise AssertionError(operator)

    def _evaluate_leaf(self, license_id: str, context: str) -> tuple[str, str, str]:
        normalized = license_id.strip()
        upper = normalized.upper()
        unknown_values = {value.upper() for value in self.rules.get("unknown", [])}
        if not normalized or upper in unknown_values:
            status = "warn" if self.unknown_policy == "warn" else "deny"
            return status, "UNKNOWN_LICENSE", "license metadata is missing or non-standard"

        for pattern in self.rules.get("deny", []):
            if fnmatch.fnmatchcase(normalized.lower(), pattern.lower()):
                return "deny", "DENIED_LICENSE", f"{normalized} is denied in every context"

        if normalized in self.rules.get("allow", []):
            if normalized == "Apache-2.0" and context in {"SHIP", "PUBLISH"}:
                return "info", "APACHE_NOTICE", "propagate applicable Apache NOTICE content"
            return "allow", "ALLOWED_LICENSE", f"{normalized} is allowed"

        for pattern, contexts in self.rules.get("conditional", {}).items():
            if fnmatch.fnmatchcase(normalized.lower(), pattern.lower()):
                rule = contexts[context]
                return rule["status"], rule["code"], rule["detail"]

        status = "warn" if self.unknown_policy == "warn" else "deny"
        return status, "UNCLASSIFIED_LICENSE", f"{normalized} is not classified by policy"


def load_exceptions(path: Path) -> list[dict[str, str]]:
    data = read_json(path)
    if not isinstance(data, list):
        raise PolicyError("exceptions file must contain a JSON array")
    required = {"package", "version", "license", "reason", "approved_by", "review_by"}
    output: list[dict[str, str]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not required.issubset(item):
            missing = sorted(required - set(item if isinstance(item, dict) else {}))
            raise PolicyError(f"exception {index} is missing required fields: {', '.join(missing)}")
        try:
            dt.date.fromisoformat(str(item["review_by"]))
        except ValueError as exc:
            raise PolicyError(f"exception {index} has invalid review_by date") from exc
        output.append({key: str(value) for key, value in item.items()})
    return output


def exception_for(
    exceptions: list[dict[str, str]], package: str, version: str, expression: str
) -> tuple[dict[str, str] | None, str]:
    try:
        expression_leaves = leaves(ExpressionParser(expression).parse())
    except ValueError:
        expression_leaves = set()
    for item in exceptions:
        if item["package"].lower() != package.lower() or item["version"] != version:
            continue
        if item["license"] != expression and item["license"] not in expression_leaves:
            continue
        if dt.date.fromisoformat(item["review_by"]) < dt.date.today():
            return item, "expired"
        if item["license"] in expression_leaves and item["license"] != expression:
            return item, "election"
        return item, "exception"
    return None, ""


def tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [item for item in result.stdout.split("\0") if item]


def vendored_findings(repo_root: Path, context: str) -> list[Finding]:
    files = tracked_files(repo_root)
    roots: set[str] = set()
    for filename in files:
        parts = Path(filename).parts
        for index, part in enumerate(parts[:-1]):
            if part.lower() in VENDOR_DIR_NAMES:
                roots.add(Path(*parts[: index + 1]).as_posix())
                break

    findings: list[Finding] = []
    for root in sorted(roots):
        source_files = [name for name in files if name.startswith(root + "/") and Path(name).suffix in SOURCE_EXTENSIONS]
        if not source_files:
            continue
        has_license = any(
            name.startswith(root + "/") and Path(name).name.lower() in LICENSE_FILENAMES
            for name in files
        )
        if not has_license:
            findings.append(
                Finding(
                    code="VENDORED_LICENSE_MISSING",
                    status="deny",
                    context=context,
                    scope="runtime",
                    ecosystem="vendored",
                    package=root,
                    version="tracked-source",
                    license_expression="NOASSERTION",
                    source=root,
                    detail="vendored source directory has no tracked LICENSE or NOTICE file",
                ).finalize()
            )

    gitmodules = repo_root / ".gitmodules"
    if gitmodules.is_file():
        for submodule_path in re.findall(
            r"^\s*path\s*=\s*(.+?)\s*$", gitmodules.read_text(encoding="utf-8"), re.MULTILINE
        ):
            prefix = submodule_path.strip().rstrip("/")
            has_license = any(
                name.startswith(prefix + "/") and Path(name).name.lower() in LICENSE_FILENAMES
                for name in files
            )
            if not has_license:
                findings.append(
                    Finding(
                        code="SUBMODULE_LICENSE_UNVERIFIED",
                        status="deny",
                        context=context,
                        scope="runtime",
                        ecosystem="git-submodule",
                        package=prefix,
                        version="gitlink",
                        license_expression="NOASSERTION",
                        source=".gitmodules",
                        detail="submodule license is not present in the parent repository",
                    ).finalize()
                )
    return findings


def current_commit(repo_root: Path) -> str:
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def make_report(
    commit: str,
    classified: list[Finding],
    new_violations: list[Finding],
    baselined: list[Finding],
    warnings: list[Finding],
    dev_findings: list[Finding],
    exceptions: list[Finding],
) -> str:
    lines = [
        "# Dependency licence policy",
        "",
        f"Scanned commit `{commit}`.",
        "",
        "| Result | Count |",
        "|---|---:|",
        f"| Dependencies classified | {len(classified)} |",
        f"| New violations | {len(new_violations)} |",
        f"| Baselined violations | {len(baselined)} |",
        f"| Runtime warnings/obligations | {len(warnings)} |",
        f"| Dev-only findings (non-blocking) | {len(dev_findings)} |",
        f"| Active exceptions | {len(exceptions)} |",
        "",
    ]

    def section(title: str, values: list[Finding]) -> None:
        if not values:
            return
        lines.extend(
            [
                f"## {title}",
                "",
                "| Package | Version | Licence | Context | Source | Detail |",
                "|---|---|---|---|---|---|",
            ]
        )
        for finding in sorted(values, key=lambda item: (item.package, item.version, item.source))[:100]:
            detail = finding.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{finding.package}` | `{finding.version}` | `{finding.license_expression}` "
                f"| {finding.context} | `{finding.source}` | {detail} |"
            )
        if len(values) > 100:
            lines.append(f"\n{len(values) - 100} additional findings are in the JSON artifact.")
        lines.append("")

    section("New violations", new_violations)
    section("Baselined violations", baselined)
    section("Runtime warnings and obligations", warnings)
    section("Dev-only findings", dev_findings)
    section("Active exceptions", exceptions)
    lines.extend(
        [
            "The committed baseline suppresses only unchanged historical violations. "
            "New package versions receive new fingerprints and are evaluated again.",
            "",
            "Automated classification is not legal advice.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).resolve()
    config = read_json(repo_root / args.config)
    if config.get("schema_version") != 1:
        raise PolicyError("unsupported .license-policy.json schema_version")
    policy = Policy(config)
    scan = read_json(repo_root / args.scan)
    if not isinstance(scan.get("results"), list):
        raise PolicyError("OSV-Scanner output is missing results")

    exceptions_data = load_exceptions(repo_root / args.exceptions)
    first_party = collect_first_party_names(repo_root)
    first_party.update(
        normalize_name(item, "PyPI") for item in config.get("first_party_packages", [])
    )
    scope_index = ScopeIndex(repo_root, set(config.get("dev_groups", [])))
    ignored_sources = set(config.get("ignore_sources", []))
    seen_sources: set[str] = set()
    classified: list[Finding] = []
    exception_findings: list[Finding] = []

    for result in scan["results"]:
        source = clean_source_path(str(result.get("source", {}).get("path", "")), repo_root)
        if source in ignored_sources:
            continue
        seen_sources.add(source)
        context = policy.context_for(source)
        for item in result.get("packages", []):
            package = item.get("package", {})
            ecosystem = str(package.get("ecosystem", "unknown"))
            name = str(package.get("name", ""))
            version = str(package.get("version", ""))
            if not name or not version:
                continue
            normalized_name = normalize_name(name, ecosystem)
            if normalized_name in first_party:
                continue
            key = PackageKey(ecosystem, normalized_name, version)
            scope = scope_index.scope(source, key)
            license_values = item.get("licenses") or ["UNKNOWN"]
            expression = " AND ".join(f"({value})" for value in license_values)
            if len(license_values) == 1:
                expression = str(license_values[0])

            exception, exception_state = exception_for(
                exceptions_data, normalized_name, version, expression
            )
            if exception_state == "expired":
                finding = Finding(
                    code="EXPIRED_EXCEPTION",
                    status="deny",
                    context=context,
                    scope=scope,
                    ecosystem=ecosystem,
                    package=normalized_name,
                    version=version,
                    license_expression=expression,
                    source=source,
                    detail=f"exception expired on {exception['review_by']}",
                ).finalize()
            elif exception_state == "exception":
                finding = Finding(
                    code="ACTIVE_EXCEPTION",
                    status="info",
                    context=context,
                    scope=scope,
                    ecosystem=ecosystem,
                    package=normalized_name,
                    version=version,
                    license_expression=expression,
                    source=source,
                    detail=f"approved by {exception['approved_by']} until {exception['review_by']}",
                ).finalize()
                exception_findings.append(finding)
            else:
                elected = exception["license"] if exception_state == "election" and exception else ""
                status, code, detail = policy.evaluate_expression(expression, context, elected)
                finding = Finding(
                    code=code,
                    status=status,
                    context=context,
                    scope=scope,
                    ecosystem=ecosystem,
                    package=normalized_name,
                    version=version,
                    license_expression=expression,
                    source=source,
                    detail=detail,
                ).finalize()
                if exception_state == "election":
                    exception_findings.append(finding)
            classified.append(finding)

    missing_sources = [path for path in config.get("required_scan_paths", []) if path not in seen_sources]
    for source in missing_sources:
        classified.append(
            Finding(
                code="LOCKFILE_NOT_SCANNED",
                status="deny",
                context=policy.context_for(source),
                scope="runtime",
                ecosystem="scanner",
                package=source,
                version="current",
                license_expression="NOASSERTION",
                source=source,
                detail="required lockfile was not present in OSV-Scanner output",
            ).finalize()
        )

    classified.extend(vendored_findings(repo_root, config["default_context"]))
    commit = current_commit(repo_root)
    baseline_path = repo_root / args.baseline

    if args.write_baseline:
        denial_findings = [
            finding
            for finding in classified
            if finding.status == "deny" and finding.scope != "dev"
        ]
        write_json(
            baseline_path,
            {
                "schema_version": 1,
                "generated_from_commit": commit,
                "findings": [finding.as_dict() for finding in denial_findings],
            },
        )

    baseline = read_json(baseline_path)
    if baseline.get("schema_version") != 1:
        raise PolicyError("unsupported baseline schema_version")
    baseline_fingerprints = {
        item.get("fingerprint", "") for item in baseline.get("findings", []) if isinstance(item, dict)
    }

    new_violations: list[Finding] = []
    baselined: list[Finding] = []
    warnings: list[Finding] = []
    dev_findings: list[Finding] = []
    for finding in classified:
        if finding.scope == "dev":
            if finding.status in {"info", "warn", "deny"} and finding.code != "ACTIVE_EXCEPTION":
                dev_findings.append(finding)
        elif finding.status == "deny":
            if finding.fingerprint in baseline_fingerprints:
                baselined.append(finding)
            else:
                new_violations.append(finding)
        elif finding.status in {"warn", "info"} and finding.code != "ACTIVE_EXCEPTION":
            warnings.append(finding)

    results = {
        "schema_version": 1,
        "scanned_commit": commit,
        "dependencies": [finding.as_dict() for finding in classified],
        "new_violations": [finding.as_dict() for finding in new_violations],
        "baselined_violations": [finding.as_dict() for finding in baselined],
        "warnings": [finding.as_dict() for finding in warnings],
        "dev_only_findings": [finding.as_dict() for finding in dev_findings],
        "exceptions": [finding.as_dict() for finding in exception_findings],
    }
    write_json(repo_root / args.results_json, results)
    report = make_report(
        commit,
        classified,
        new_violations,
        baselined,
        warnings,
        dev_findings,
        exception_findings,
    )
    (repo_root / args.report).write_text(report, encoding="utf-8")
    print(report)
    return 1 if new_violations else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--scan", default=".license-scan/osv.json")
    parser.add_argument("--config", default=".license-policy.json")
    parser.add_argument("--baseline", default=".license-policy-baseline.json")
    parser.add_argument("--exceptions", default=".license-policy-exceptions.json")
    parser.add_argument("--report", default=".license-scan/report.md")
    parser.add_argument("--results-json", default=".license-scan/results.json")
    parser.add_argument("--write-baseline", action="store_true")
    return parser


def main() -> int:
    try:
        return evaluate(build_parser().parse_args())
    except (PolicyError, subprocess.CalledProcessError) as exc:
        print(f"license-policy: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
