"""Tiny pytree helpers for observations that may be arrays, dicts or tuples."""
from typing import Any, Callable, Sequence
import torch

type Tree = Any


def tree_map(fn: Callable[[Any], Any], tree: Tree) -> Tree:
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, v) for v in tree)
    return fn(tree)


def tree_stack(trees: Sequence[Tree], dim: int = 0) -> Tree:
    """Stack a list of same-structured trees of tensors leaf-wise."""
    first = trees[0]
    if isinstance(first, dict):
        return {k: tree_stack([t[k] for t in trees], dim) for k in first}
    if isinstance(first, tuple):
        return tuple(tree_stack([t[i] for t in trees], dim) for i in range(len(first)))
    return torch.stack(list(trees), dim)


def tree_set(tree: Tree, index: int, value: Tree) -> None:
    """In place: tree[index] = value, leaf-wise (for batched numpy / torch leaves)."""
    if isinstance(tree, dict):
        for k in tree:
            tree_set(tree[k], index, value[k])
    elif isinstance(tree, tuple):
        for leaf, v in zip(tree, value):
            tree_set(leaf, index, v)
    else:
        tree[index] = value
