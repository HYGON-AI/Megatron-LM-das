import re
import os
from abc import ABC, abstractmethod
from typing import List, Union
from torch.utils.cpp_extension import load
from torch.library import Library
import dcu_megatron

AS_LIBRARY = Library("dcu_megatron", "DEF")


class DCUMegatronOpBuilder(ABC):
    _loaded_ops = {}

    def __init__(self, name):
        self.name = name

    def get_absolute_paths(self, paths):
        dcu_megatron_path = os.path.abspath(os.path.dirname(dcu_megatron.__file__))
        return [os.path.join(dcu_megatron_path, path) for path in paths]

    def register_op_proto(self, op_proto: Union[str, List[str]]):
        if isinstance(op_proto, str):
            op_proto = [op_proto]
        for proto in op_proto:
            AS_LIBRARY.define(proto)

    @abstractmethod
    def sources(self):
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
                         verbose=verbose)
        __class__._loaded_ops[self.name] = op_module

        return op_module
