# Some of this code was adopted from https://gitcode.com/Ascend/MindSpeed
import importlib
import inspect
import sys
import types
import warnings
from typing import Dict, List, Union


def get_func_name(func):
    if isinstance(func, str):
        return func
    return '.'.join((func.__module__, func.__qualname__))


def dummy_function_wrapper(func_name):
    def dummy_function(*args, **kwargs):
        raise RuntimeError('function {} no exist'.format(func_name))

    return dummy_function


def get_inner_func(func, inner_func_name):
    for c in func.__code__.co_consts:
        if isinstance(c, types.CodeType) and c.co_name == inner_func_name:
            return c

    raise RuntimeError(f"Failed to retrieve inner function {inner_func_name}")


class Patch:
    def __init__(
        self,
        orig_func_or_cls_name,
        new_func_or_cls,
        create_dummy,
        apply_wrapper=False,
        remove_origin_wrappers=False,
        patch_inner_func=False,
        inner_func_name=None,
    ):
        split_name = orig_func_or_cls_name.rsplit('.', 1)
        if len(split_name) == 1:
            self.orig_module_name, self.orig_func_or_cls_name = orig_func_or_cls_name, None
        else:
            self.orig_module_name, self.orig_func_or_cls_name = split_name
        self.orig_module = None
        self.orig_func_or_cls = None

        self.patch_func_or_cls = None
        self.patch_inner_funcs: Dict[str, types.CodeType] = dict()    # inner funcs that will be replaced
        self.wrappers = []                                            # new wrappers
        self.remove_origin_wrappers = False
        if (
            new_func_or_cls is None
            and not remove_origin_wrappers
        ):
            new_func_or_cls = dummy_function_wrapper(orig_func_or_cls_name)

        self.set_patch_func(
            new_func_or_cls,
            apply_wrapper=apply_wrapper,
            remove_origin_wrappers=remove_origin_wrappers,
            patch_inner_func=patch_inner_func,
            inner_func_name=inner_func_name,
        )
        self.is_applied = False
        self.create_dummy = create_dummy

    @property
    def orig_func_or_cls_id(self):
        return id(self.orig_func_or_cls)

    @property
    def patch_func_id(self):
        return id(self.patch_func_or_cls)

    @staticmethod
    def remove_wrappers(module, func_name, func):
        wrappers = []

        if (
            module.__dict__
            and func_name in module.__dict__
            and isinstance(module.__dict__[func_name], (staticmethod, classmethod))
        ):
            descriptor = module.__dict__[func_name]
            wrappers.append({
                "type": type(descriptor),
                "wrapper": descriptor,
            })
            func = descriptor.__func__

        while True:
            if hasattr(func, "__wrapped__") and func.__wrapped__ is not None:
                wrappers.append({
                    "type": "__wrapped__",
                    "wrapper": func,
                })
                func = func.__wrapped__

            elif hasattr(func, "__closure__") and func.__closure__ is not None:
                closure_func = None

                for cell in func.__closure__:
                    cell_value = cell.cell_contents
                    if callable(cell_value):
                        closure_func = cell_value
                        break

                if closure_func is None:
                    break

                wrappers.append({
                    "type": "__closure__",
                    "wrapper": func,
                })
                func = closure_func

            else:
                break

        return func, wrappers

    @staticmethod
    def add_origin_wrappers(func, wrappers: list):
        for item in reversed(wrappers):
            if item["type"] in (staticmethod, classmethod):
                func = item["type"](func)
            elif item["type"] == "__wrapped__":
                wrapper = item["wrapper"]
                wrapper.__wrapped__ = func
                func = wrapper
            elif item["type"] == "__closure__":
                # Closure wrappers generally cannot be safely restored in-place;
                # they can only be reused from the original wrapper
                func = item["wrapper"]

        return func

    def set_patch_func(
        self,
        new_func_or_cls=None,
        force_patch=False,
        apply_wrapper=False,
        remove_origin_wrappers=False,
        patch_inner_func=False,
        inner_func_name=None
    ):
        """
            patch_inner_func: If true, func that needs patching is an inner one
        """
        if remove_origin_wrappers:
            self.remove_origin_wrappers = True
        else:
            assert new_func_or_cls is not None

        if patch_inner_func:
            assert not apply_wrapper, "apply_wrapper should be False"
            assert new_func_or_cls is not None and inner_func_name is not None
            if (
                self.patch_inner_funcs.get(inner_func_name, None) is not None
                and id(new_func_or_cls) != id(self.patch_inner_funcs[inner_func_name])
            ):
                raise RuntimeError('the patch of {} exist !'.format(inner_func_name))
            self.patch_inner_funcs[inner_func_name] = new_func_or_cls
            return

        if new_func_or_cls is None:
            return

        if (
            apply_wrapper
            or (hasattr(new_func_or_cls, '__name__') and new_func_or_cls.__name__.endswith(('wrapper', 'decorator')))
        ):
            for wrapper in self.wrappers:
                if id(wrapper) == id(new_func_or_cls):
                    raise RuntimeError(f"wrapper {getattr(new_func_or_cls, '__name__')} has already been applied")
            self.wrappers.append(new_func_or_cls)
        else:
            if (
                self.patch_func_or_cls
                and not force_patch
                and id(new_func_or_cls) != id(self.patch_func_or_cls)
            ):
                raise RuntimeError('the patch of {} exist !'.format(self.orig_func_or_cls_name))
            self.patch_func_or_cls = new_func_or_cls
        self.is_applied = False

    def apply_patch(self):
        if self.is_applied:
            return

        self.orig_module, self.orig_func_or_cls = Patch.parse_path(self.orig_module_name, self.orig_func_or_cls_name, self.create_dummy)

        final_patch_func_or_cls = self.orig_func_or_cls
        if self.patch_func_or_cls is not None:
            final_patch_func_or_cls = self.patch_func_or_cls

        # remove original wrappers
        if self.remove_origin_wrappers or self.patch_inner_funcs:
            final_patch_func_or_cls, origin_wrappers = self.remove_wrappers(
                self.orig_module,
                self.orig_func_or_cls_name,
                final_patch_func_or_cls
            )

        # replace inner funcs
        if self.patch_inner_funcs:
            new_consts = list()
            for c in final_patch_func_or_cls.__code__.co_consts:
                if isinstance(c, types.CodeType) and c.co_name in self.patch_inner_funcs:
                    inner_func = self.patch_inner_funcs.pop(c.co_name)
                    new_consts.append(inner_func if isinstance(inner_func, types.CodeType) else inner_func.__code__)
                else:
                    new_consts.append(c)

            for inner_func_name in self.patch_inner_funcs:
                warnings.warn(f"inner func {inner_func_name} of {self.orig_func_or_cls_name} has NOT been replaced")
            setattr(final_patch_func_or_cls, "__code__", final_patch_func_or_cls.__code__.replace(co_consts=tuple(new_consts)))

            if not self.remove_origin_wrappers:
                final_patch_func_or_cls = self.add_origin_wrappers(final_patch_func_or_cls, origin_wrappers)

        # add new wrappers
        for wrapper in self.wrappers:
            final_patch_func_or_cls = wrapper(final_patch_func_or_cls)

        # patch funcs
        if self.orig_func_or_cls_name is not None:
            setattr(self.orig_module, self.orig_func_or_cls_name, final_patch_func_or_cls)
        for key, value in sys.modules.copy().items():
            if (
                self.orig_func_or_cls_name is not None
                and hasattr(value, self.orig_func_or_cls_name)
                and id(getattr(value, self.orig_func_or_cls_name)) == self.orig_func_or_cls_id
            ):
                setattr(value, self.orig_func_or_cls_name, final_patch_func_or_cls)

        self.is_applied = True
        self.final_patch_func_or_cls = final_patch_func_or_cls

    @staticmethod
    def parse_path(module_path, function_name, create_dummy):
        from importlib.machinery import ModuleSpec
        modules = module_path.split('.')
        for i in range(1, len(modules) + 1):
            parent = '.'.join(modules[:i - 1])
            path = '.'.join(modules[:i])
            try:
                importlib.import_module(path)
            except ModuleNotFoundError as e:
                if not parent or not hasattr(importlib.import_module(parent), modules[i - 1]):
                    if not create_dummy:
                        raise ModuleNotFoundError(e) from e
                    sys.modules[path] = types.ModuleType(path)
                    sys.modules[path].__file__ = 'hcu_megatron.dummy_module.py'
                    sys.modules[path].__spec__ = ModuleSpec(path, None)
                    if parent:
                        setattr(importlib.import_module(parent), modules[i - 1], sys.modules[path])
                else:
                    module = getattr(importlib.import_module(parent), modules[i - 1])
                    if hasattr(module, function_name):
                        return module, getattr(module, function_name)
                    elif create_dummy:
                        return module, dummy_function_wrapper(function_name)
                    else:
                        raise RuntimeError('no exist {} of {}'.format(function_name, module))

        if function_name is not None and not hasattr(sys.modules[module_path], function_name):
            assert create_dummy, f"{function_name} of {module_path} does not exist"
            setattr(sys.modules[module_path], function_name, None)
        return sys.modules[module_path], getattr(sys.modules[module_path], function_name) if function_name is not None else None

    def remove_patch_wrappers(self, wrapper_names: Union[str, List[str]] = None):
        if wrapper_names is None:
            self.wrappers.clear()
            return

        if isinstance(wrapper_names, str):
            wrapper_names = [wrapper_names]
        for name in wrapper_names:
            i = 0
            while i < len(self.wrappers):
                if self.wrappers[i].__name__ == name:
                    self.wrappers.pop(i)
                else:
                    i += 1

    def remove_patch(self):
        for key, value in sys.modules.copy().items():
            if 'hcu_megatron' in key or 'torch.classes' == key:
                continue

            if inspect.isclass(self.orig_module) and hasattr(value, self.orig_module_name.split('.')[-1]):
                value = getattr(value, self.orig_module_name.split('.')[-1])

            if self.orig_func_or_cls_name is not None and hasattr(value, self.orig_func_or_cls_name) \
                    and id(getattr(value, self.orig_func_or_cls_name)) == id(self.final_patch_func_or_cls):
                setattr(value, self.orig_func_or_cls_name, self.orig_func_or_cls)
        self.patch_func_or_cls = None
        self.final_patch_func_or_cls = None
        self.patch_inner_funcs.clear()
        self.is_applied = False


class MegatronPatchesManager:
    patches_info = {}

    @staticmethod
    def register_patch(
        orig_func_or_cls_name,
        new_func_or_cls=None,
        force_patch=False,
        create_dummy=False,
        apply_wrapper=False,
        remove_origin_wrappers=False,
        patch_inner_func=False,
        inner_func_name=None,
    ):
        if orig_func_or_cls_name not in MegatronPatchesManager.patches_info:
            MegatronPatchesManager.patches_info[orig_func_or_cls_name] = Patch(
                orig_func_or_cls_name,
                new_func_or_cls,
                create_dummy,
                apply_wrapper=apply_wrapper,
                remove_origin_wrappers=remove_origin_wrappers,
                patch_inner_func=patch_inner_func,
                inner_func_name=inner_func_name,
            )
        else:
            MegatronPatchesManager.patches_info.get(orig_func_or_cls_name).set_patch_func(
                new_func_or_cls,
                force_patch,
                apply_wrapper=apply_wrapper,
                remove_origin_wrappers=remove_origin_wrappers,
                patch_inner_func=patch_inner_func,
                inner_func_name=inner_func_name,
            )

    @staticmethod
    def register_cls_funcs(orig_class, new_funcs: list = None, create_dummy=False):
        if not orig_class.endswith("."):
            orig_class += "."

        for new_func in new_funcs:
            assert hasattr(new_func, '__name__') and not new_func.__name__.endswith(('wrapper', 'decorator'))

            orig_func_name = orig_class + new_func.__name__
            MegatronPatchesManager.register_patch(orig_func_name, new_func_or_cls=new_func, create_dummy=create_dummy)

    @staticmethod
    def apply_patches():
        for patch in MegatronPatchesManager.patches_info.values():
            patch.apply_patch()

    @staticmethod
    def remove_patches():
        for patch in MegatronPatchesManager.patches_info.values():
            patch.remove_patch()
            patch.remove_patch_wrappers()
