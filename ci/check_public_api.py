"""Enforce explicit public API contracts for porki modules."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "porki"


class ApiContractError(RuntimeError):
    """Raised when module export contract validation fails."""


def _extract_public_candidates(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                names.add(node.name)
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    ident = target.id
                    if ident.isupper() and not ident.startswith("_"):
                        names.add(ident)
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            ident = node.target.id
            if ident.isupper() and not ident.startswith("_"):
                names.add(ident)
    return names


def _extract___all__(tree: ast.Module) -> set[str] | None:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                try:
                    value = ast.literal_eval(node.value)
                except (ValueError, SyntaxError) as exc:
                    raise ApiContractError("__all__ must be a static literal list/tuple") from exc
                if not isinstance(value, (list, tuple)):
                    raise ApiContractError("__all__ must be a list or tuple")
                entries: set[str] = set()
                for item in value:
                    if not isinstance(item, str):
                        raise ApiContractError("__all__ entries must be strings")
                    entries.add(item)
                return entries
    return None


def _validate_module(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    candidates = _extract_public_candidates(tree)
    exported = _extract___all__(tree)
    errors: list[str] = []

    if exported is None:
        errors.append(f"{path}: missing __all__")
        return errors

    unexported_public = sorted(candidates - exported)
    if unexported_public:
        errors.append(
            f"{path}: public symbols missing from __all__: {', '.join(unexported_public)}"
        )

    unknown_exports = sorted(exported - candidates)
    if unknown_exports:
        errors.append(
            f"{path}: __all__ exports unknown module symbols: {', '.join(unknown_exports)}"
        )

    return errors


def main() -> int:
    modules = sorted(path for path in PACKAGE_DIR.glob("*.py") if path.name != "__init__.py")
    errors: list[str] = []
    for module in modules:
        errors.extend(_validate_module(module))

    if errors:
        print("Public API contract check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Public API contract check passed ({len(modules)} modules)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
