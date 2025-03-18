import importlib
import os
from itertools import chain
from types import ModuleType
from typing import Any
from transformers.utils.import_utils import _is_package_available

_deps = [
    ('deepspeed', '_deepspeed_available'),
    ('diffusers', '_diffusers_available'),
    ('llm_blender', '_llm_blender_available'),
    ('mergekit', '_mergekit_available'),
    ('rich', '_rich_available'),
    ('unsloth', '_unsloth_available'),
    ('vllm', '_vllm_available'),
]
for _p, _v in _deps:
    globals()[_v] = _is_package_available(_p)

def is_deepspeed_available() -> bool:
    return _deepspeed_available

def is_diffusers_available() -> bool:
    return _diffusers_available

def is_llm_blender_available() -> bool:
    return _llm_blender_available

def is_mergekit_available() -> bool:
    return _mergekit_available

def is_vllm_available() -> bool:
    return _vllm_available

class _LazyModule(ModuleType):
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_module_map = {}
        for k, v in import_structure.items():
            for x in v:
                self._class_module_map[x] = k
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._extra_objs = {} if extra_objects is None else extra_objects
        self._mod_name = name
        self._import_structure = import_structure

    def __dir__(self):
        r = super().__dir__()
        for a in self.__all__:
            if a not in r:
                r.append(a)
        return r

    def __getattr__(self, name: str) -> Any:
        if name in self._extra_objs:
            return self._extra_objs[name]
        if name in self._modules:
            val = self._fetch_module(name)
        elif name in self._class_module_map:
            m = self._fetch_module(self._class_module_map[name])
            val = getattr(m, name)
        else:
            raise AttributeError(f'模块 {self.__name__} 没有属性 {name}')
        setattr(self, name, val)
        return val

    def _fetch_module(self, module_name: str):
        try:
            return importlib.import_module('.' + module_name, self.__name__)
        except Exception as _err:
            raise RuntimeError(f'无法导入 {self.__name__}.{module_name}，因以下错误（请查看其追溯信息）:\n{_err}') from _err

    def __reduce__(self):
        return (self.__class__, (self._mod_name, self.__file__, self._import_structure))

class OptionalDependencyNotAvailable(BaseException):
    pass
