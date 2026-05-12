"""
compat.py – Parches de compatibilidad de dependencias externas.

DEBE importarse como primer módulo en cualquier script de entrada
(train.py, evaluate.py) antes de importar torch_geometric.

Contexto
--------
PyTorch Geometric usa `torch_geometric.inspector.type_repr` para
introspección de tipos en Python 3.13+. La función falla con
`AttributeError: '_name'` al encontrar `typing.Union` internamente.
El parche reemplaza la función con una versión defensiva que captura
ese error y devuelve el string "Union" como fallback.
"""

from __future__ import annotations


def _apply_pyg_union_patch() -> None:
    """Aplica el parche de type_repr sobre torch_geometric.inspector."""
    try:
        import torch_geometric.inspector as _pyg_inspector

        _original = _pyg_inspector.type_repr

        def _safe_type_repr(obj, _globals=None):
            try:
                return _original(obj, _globals)
            except AttributeError as exc:
                if "'_name'" in str(exc):
                    return "Union"
                raise

        _pyg_inspector.type_repr = _safe_type_repr
    except Exception:
        # Si PyG no está instalado o el inspector tiene otra forma,
        # no hacemos nada —el error aparecerá en el import original.
        pass


_apply_pyg_union_patch()
