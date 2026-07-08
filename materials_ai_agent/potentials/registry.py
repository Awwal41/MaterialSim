"""Runtime registry + auto-resolution of potential providers."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set

from .base import PotentialNotSupported, PotentialProvider

# Preference order used when kind == 'auto': more transferable / accurate first,
# but only among providers that are actually available and applicable.
_AUTO_ORDER = [
    "emt",
    "eam",
    "tersoff",
    "meam",
    "reaxff",
    "mace",
    "chgnet",
    "m3gnet",
    "opls",
    "gaff",
    "openff",
    "lj",
]

_PROVIDERS: Dict[str, Callable[..., PotentialProvider]] = {}


def register_potential(kind: str, factory: Callable[..., PotentialProvider]) -> None:
    _PROVIDERS[kind.lower()] = factory


def list_potentials() -> List[str]:
    return sorted(_PROVIDERS)


def _make(kind: str, spec=None) -> Optional[PotentialProvider]:
    factory = _PROVIDERS.get(kind.lower())
    if factory is None:
        return None
    try:
        return factory(spec)
    except TypeError:
        return factory()


def resolve_potential(
    kind: str,
    elements: Set[str],
    *,
    engine: Optional[str] = None,
    bonded: bool = False,
    spec=None,
) -> PotentialProvider:
    """Return a usable provider for *kind* + chemistry, or raise a clear error.

    ``kind='auto'`` scans the preference order and returns the first provider
    that is available, supports the chemistry, and (if given) targets *engine*.
    """
    kind = (kind or "auto").lower()

    def _ok(p: PotentialProvider) -> bool:
        if not p.available():
            return False
        if engine and p.engines and engine not in p.engines:
            return False
        try:
            return p.supports(elements, bonded=bonded)
        except Exception:
            return False

    if kind != "auto":
        provider = _make(kind, spec)
        if provider is None:
            raise PotentialNotSupported(
                f"No provider registered for potential '{kind}'. "
                f"Registered: {list_potentials()}."
            )
        if not provider.available():
            raise PotentialNotSupported(
                f"Potential '{kind}' is registered but its backend is not "
                f"installed/configured."
            )
        if engine and provider.engines and engine not in provider.engines:
            raise PotentialNotSupported(
                f"Potential '{kind}' does not target engine '{engine}' "
                f"(supported: {sorted(provider.engines)})."
            )
        if not provider.supports(elements, bonded=bonded):
            raise PotentialNotSupported(
                f"Potential '{kind}' does not support "
                f"{{{', '.join(sorted(elements))}}}"
                + (" as a bonded system." if bonded else ".")
            )
        return provider

    for candidate in _AUTO_ORDER:
        provider = _make(candidate, spec)
        if provider is not None and _ok(provider):
            return provider
    # anything registered outside the ordered list
    for name in list_potentials():
        provider = _make(name, spec)
        if provider is not None and _ok(provider):
            return provider

    raise PotentialNotSupported(
        "No available potential can model "
        f"{{{', '.join(sorted(elements))}}}"
        + (" as a bonded system" if bonded else "")
        + ". Install an MLIP (e.g. MACE/CHGNet) for arbitrary chemistry, "
        "or provide a classical potential file for LAMMPS."
    )
