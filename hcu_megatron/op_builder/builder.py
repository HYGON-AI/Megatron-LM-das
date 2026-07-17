import re
import os
from abc import ABC, abstractmethod
from typing import List, Union
from torch.utils.cpp_extension import load, _import_module_from_library
from torch.library import Library
from megatron.training.utils import print_rank_0

import hcu_megatron

AS_LIBRARY = Library("hcu_megatron", "DEF")


class HCUMegatronOpBuilder(ABC):
    _loaded_ops = {}

    def __init__(self, name):
        self.name = name

    def get_absolute_paths(self, paths):
        hcu_megatron_path = os.path.abspath(os.path.dirname(hcu_megatron.__file__))
        return [os.path.join(hcu_megatron_path, path) for path in paths]

    def register_op_proto(self, op_proto: Union[str, List[str]]):
        if isinstance(op_proto, str):
            op_proto = [op_proto]
        for proto in op_proto:
            AS_LIBRARY.define(proto)

    @abstractmethod
    def sources(self):
        ...

    @abstractmethod
    def compiled_files(self):
        ...

    def include_paths(self):
        return None

    def cxx_args(self):
        args = ['-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack', '-fPIC', '-pie',
                '-s', '-fvisibility=hidden', '-D_FORTIFY_SOURCE=2', '-O2']
        return args

    def extra_ldflags(self):
        return None

    def load(self, verbose=True):
        if self.name in __class__._loaded_ops:
            return __class__._loaded_ops[self.name]

        op_module = load(name=self.name,
                         sources=self.get_absolute_paths(self.sources()),
                         extra_include_paths=None,
                         extra_cflags=self.cxx_args(),
                         extra_ldflags=None,
                         build_directory=os.path.dirname(self.get_absolute_paths(self.sources())[0]),
                         verbose=verbose)
        __class__._loaded_ops[self.name] = op_module

        return op_module

    def import_module_from_library(self):
        if self.name in __class__._loaded_ops:
            return __class__._loaded_ops[self.name]

        op_module = _import_module_from_library(
            module_name=self.name,
            path=os.path.dirname(self.get_absolute_paths(self.compiled_files())[0]),
            is_python_module=True
        )

        __class__._loaded_ops[self.name] = op_module
        return op_module

    def get_module(self):
        try:
            print_rank_0("Start reading the compiled so.")
            op_module = self.import_module_from_library()
        except Exception as e:
            print_rank_0("Failed to read the compiled so, recompile.")
            op_module = self.load()

        return op_module
