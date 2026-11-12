"""Names kept for compatibility after a renaming.

The toolbox follows the wording of the paper. When a revision of the paper drops a name, the function, attribute,
argument or key that carried it is renamed here as well, and the old spelling keeps working for a transition period
with a DeprecationWarning that states the new one. Nothing in this module changes a computed value.
"""
from __future__ import annotations

import functools
import warnings
from typing import Callable, Dict


def _warn(old: str, new: str, stacklevel: int = 3) -> None:
    warnings.warn(f"{old} is deprecated; use {new}", DeprecationWarning, stacklevel=stacklevel)


def module_getattr(module: str, renamed: Dict[str, str], namespace: dict) -> Callable[[str], object]:
    """A module-level __getattr__ that serves the old names of `renamed` (old -> new) with a warning."""
    def __getattr__(name: str):
        if name in renamed:
            _warn(f"{module}.{name}", f"{module}.{renamed[name]}")
            return namespace[renamed[name]]
        raise AttributeError(f"module {module!r} has no attribute {name!r}")
    return __getattr__


def renamed_arguments(**renamed: str):
    """Decorator: accept the old keyword arguments of `renamed` (old=new) with a warning."""
    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old, new in renamed.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(f"{func.__name__}() got both {old!r} (deprecated) and {new!r}")
                    _warn(f"the argument {old} of {func.__name__}", new)
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)
        return wrapper
    return decorate


def renamed_attribute(old: str, new: str) -> property:
    """A read-only property that serves the old attribute name with a warning."""
    def getter(self):
        _warn(f"{type(self).__name__}.{old}", f"{type(self).__name__}.{new}")
        return getattr(self, new)
    return property(getter, doc=f"Deprecated name of `{new}`.")


class RenamedKeys(dict):
    """A dict that still answers to the old keys of `renamed` (old -> new), with a warning."""
    renamed: Dict[str, str] = {}

    def __init__(self, *args, renamed: Dict[str, str], **kwargs):
        super().__init__(*args, **kwargs)
        self.renamed = dict(renamed)

    def __missing__(self, key):
        if key in self.renamed:
            _warn(f"the key {key!r}", repr(self.renamed[key]))
            return self[self.renamed[key]]
        raise KeyError(key)

    def get(self, key, default=None):
        if key not in self and key in self.renamed:
            return self[key]
        return super().get(key, default)


class Status(str):
    """A status string that still compares equal to its old spelling, with a warning."""
    _old: Dict[str, str] = {"off-tolerance": "off-target"}

    def __eq__(self, other):
        if isinstance(other, str) and not str.__eq__(self, other) and self._old.get(str(self)) == other:
            _warn(f"the status {other!r}", repr(str(self)))
            return True
        return str.__eq__(self, other)

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = str.__hash__
