from __future__ import annotations

import numpy as np

from typing import Iterable, TypeVar, Callable

T = TypeVar("T")
V = TypeVar("V")


def diff(input_list: Iterable[T]) -> T:
    return max(input_list) - min(input_list)


def grouped_by(
    items: list[T],
    group_key: Callable[[T], V],
    eps: float,
    eps_key: Callable[[T], float],
) -> list[list[T]]:
    items = sorted(items, key=group_key)

    groups = []
    group = []

    for item in items:
        if not group:
            group.append(item)
            continue

        if group:
            cond = abs(
                group_key(item) - np.mean([group_key(item) for item in group])
            ) < eps * np.mean([eps_key(item) for item in group])
            if cond:
                group.append(item)
            else:
                groups.append(group)
                group = [item]
    else:
        if group:
            groups.append(group)
    return groups