"""Build complete editable v2 starting sources; never imported by policy workers.

Reference algorithms are copied as source, rather than imported by the candidate.
Only the selected algorithm and its dependencies are included. The event reducer in
v2_reference.py is also copied into the resulting policy.py and is fully editable.
"""
import ast
from pathlib import Path

POLICIES = ('LRU', 'LFU', 'FIFO', 'ARC', 'LeCaR', 'WA', 'UniCache',
            'OFFSET-HEAD', 'OFFSET-TAIL', 'OPT-Online', 'SimpleHYBRID-OPT', 'NONE')
ALIASES = {'SimpleHYBRID': 'UniCache', 'UniCache-OPT': 'SimpleHYBRID-OPT'}


def _definitions(path, names):
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    pieces = []
    for node in tree.body:
        name = getattr(node, 'name', None)
        if name in names:
            start = min([node.lineno] + [x.lineno for x in getattr(node, 'decorator_list', [])])
            pieces.append(''.join(lines[start - 1:node.end_lineno]))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            pieces.append(''.join(lines[node.lineno - 1:node.end_lineno]))
    return '\n\n'.join(pieces)


def source_for_policy(name):
    name = ALIASES.get(name, name)
    if name not in POLICIES:
        raise ValueError(f'Unknown online policy {name!r}; offline OPT is host-only')
    root = Path(__file__).resolve().parents[1]
    names = {'Evictor'}
    extras = {
        'LRU': {'BlockMetaData', 'LRUEvictor'},
        'NONE': {'BlockMetaData', 'LRUEvictor'},
        'LFU': {'LFUBlockMetaData', 'LFUEvictor'},
        'FIFO': {'FIFOBlockMetaData', 'FIFOEvictor'},
        'ARC': {'ARCBlockMetaData', 'ARCHistoryMetaData', 'ARCEvictor'},
        'LeCaR': {'LeCaRBlockMetaData', 'LeCaRHistoryMetaData', 'LeCaREvictor'},
        'WA': {'WABlockMetaData', 'WAEvictor'},
        'OPT-Online': {'BlockMetaData', 'OPTEvictor'},
        'OFFSET-HEAD': {'BlockMetaData', 'OffsetEvictor'}, 'OFFSET-TAIL': {'BlockMetaData', 'OffsetEvictor'},
    }
    names |= extras.get(name, set())
    if name in ('UniCache', 'SimpleHYBRID-OPT'):
        names |= {'SimpleHYBRIDlockMetaData', 'SimpleHYBRIDEvictor',
                  '_cached_lognorm_params', '_cached_lognorm_params_v1',
                  '_cached_lognorm_prob', '_cached_lognorm_cdf', 'TASK_CONFIG',
                  '_INV_HALF_LIFE', '_PRECOMPUTED_DECAY'}
    header = '''# SPDX-License-Identifier: Apache-2.0
# Standalone editable v2 policy: algorithms and bookkeeping run in this file.
import enum, heapq, math, os, random, time
from abc import ABC, abstractmethod
from collections import OrderedDict, deque, defaultdict
from functools import lru_cache
from types import SimpleNamespace
from typing import Dict, List, Tuple, Union
import numpy as np
from scipy.stats import expon
class _QuietLogger:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
logger = _QuietLogger()
'''
    enums = _definitions(root / 'vllm/simulator/hybrid_utils.py', {'TaskType', 'BlockType'}).replace('(str, Enum)', '(str, enum.Enum)')
    wa = _definitions(root / 'vllm/simulator/wa_helper.py', {'WAHelper'}) if name == 'WA' else ''
    algorithms = _definitions(root / 'vllm/core/evictor.py', names)
    reducer = Path(__file__).with_name('v2_reference.py').read_text()
    combined = header + f'\nBASELINE = {name!r}\n' + enums + '\n' + wa + '\n' + algorithms + '\n' + reducer

    return _specialize(combined, name)


class _Specialize(ast.NodeTransformer):
    def __init__(self, name):
        self.name = name

    def visit_Name(self, node):
        if node.id == 'BASELINE' and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Constant(self.name), node)
        return node

    def visit_Attribute(self, node):
        if (self.name not in ('LRU','NONE') and isinstance(node.value, ast.Name)
                and node.value.id == 'LRUEvictor' and node.attr == 'CLEANUP_THRESHOLD'):
            return ast.copy_location(ast.Constant(50), node)
        return self.generic_visit(node)

    def _constant(self, node):
        if not any(isinstance(n, (ast.Name,ast.Call,ast.Attribute,ast.Subscript,ast.Lambda,ast.comprehension)) for n in ast.walk(node)):
            try:
                value = eval(compile(ast.Expression(node), '<constant>', 'eval'), {'__builtins__':{}})
                return ast.copy_location(ast.Constant(value), node)
            except (ValueError,TypeError,NameError,KeyError,SyntaxError):
                pass
        return node

    def visit_Compare(self, node):
        return self._constant(self.generic_visit(node))

    def visit_BoolOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op,ast.And):
            if any(isinstance(v,ast.Constant) and not v.value for v in node.values):
                return ast.copy_location(ast.Constant(False),node)
            node.values = [v for v in node.values if not(isinstance(v,ast.Constant) and v.value)]
            if not node.values: return ast.copy_location(ast.Constant(True),node)
        else:
            if any(isinstance(v,ast.Constant) and v.value for v in node.values):
                return ast.copy_location(ast.Constant(True),node)
            node.values = [v for v in node.values if not(isinstance(v,ast.Constant) and not v.value)]
            if not node.values: return ast.copy_location(ast.Constant(False),node)
        return node.values[0] if len(node.values)==1 else node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_IfExp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        if not node.body:
            node.body = [ast.Pass()]
        return node

    def visit_Assign(self, node):
        if any(isinstance(t,ast.Name) and t.id=='classes' for t in node.targets):
            return None
        return self.generic_visit(node)


def _specialize(source, name):
    classes = {'LRU':'LRUEvictor','NONE':'LRUEvictor','LFU':'LFUEvictor','FIFO':'FIFOEvictor',
               'ARC':'ARCEvictor','LeCaR':'LeCaREvictor','WA':'WAEvictor','UniCache':'SimpleHYBRIDEvictor',
               'SimpleHYBRID-OPT':'SimpleHYBRIDEvictor','OPT-Online':'OPTEvictor',
               'OFFSET-HEAD':'OffsetEvictor','OFFSET-TAIL':'OffsetEvictor'}
    source=source.replace('self.algorithm = globals()[classes[BASELINE]]()',
                          f'self.algorithm = {classes[name]}()')
    tree=ast.parse(source)
    if name not in ('OPT-Online','SimpleHYBRID-OPT'):
        tree.body=[node for node in tree.body if not(isinstance(node,ast.ClassDef) and node.name=='_KnownRequests')]
    tree=_Specialize(name).visit(tree)
    ast.fix_missing_locations(tree)
    return '# SPDX-License-Identifier: Apache-2.0\n# Complete editable standalone v2 policy; no simulator imports.\n' + ast.unparse(tree) + '\n'
