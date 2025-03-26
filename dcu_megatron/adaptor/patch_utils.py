import importlib
import sys
import types


def get_func_name(func):
    if isinstance(func, str):
        return func
    return '.'.join((func.__module__, func.__qualname__))


def dummy_function_wrapper(func_name):
    def dummy_function(*args, **kwargs):
        raise RuntimeError('function {} no exist'.format(func_name))

    return dummy_function


class Patch:
    def __init__(self, orig_func_or_cls_name, new_func_or_cls, create_dummy, apply_wrapper=False):
        split_name = orig_func_or_cls_name.rsplit('.', 1)
        if len(split_name) == 1:
            self.orig_module_name, self.orig_func_or_cls_name = orig_func_or_cls_name, None
        else:
            self.orig_module_name, self.orig_func_or_cls_name = split_name
        self.orig_module = None
        self.orig_func_or_cls = None

        self.patch_func_or_cls = None
        self.wrappers = []
        if new_func_or_cls is None:
            new_func_or_cls = dummy_function_wrapper(orig_func_or_cls_name)
        self.set_patch_func(new_func_or_cls, apply_wrapper=apply_wrapper)
        self.is_applied = False
        self.create_dummy = create_dummy

    @property
    def orig_func_or_cls_id(self):
        return id(self.orig_func_or_cls)

    @property
    def patch_func_id(self):
        return id(self.patch_func_or_cls)

    def set_patch_func(self, new_func_or_cls, force_patch=False, apply_wrapper=False):
        if (
            apply_wrapper
            or (hasattr(new_func_or_cls, '__name__') and new_func_or_cls.__name__.endswith(('wrapper', 'decorator')))
        ):
            self.wrappers.append(new_func_or_cls)
        else:
            if self.patch_func_or_cls and not force_patch:
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

        for wrapper in self.wrappers:
            final_patch_func_or_cls = wrapper(final_patch_func_or_cls)

        if self.orig_func_or_cls_name is not None:
            setattr(self.orig_module, self.orig_func_or_cls_name, final_patch_func_or_cls)
        for key, value in sys.modules.copy().items():
            if self.orig_func_or_cls_name is not None and hasattr(value, self.orig_func_or_cls_name) \
                    and id(getattr(value, self.orig_func_or_cls_name)) == self.orig_func_or_cls_id:
                setattr(value, self.orig_func_or_cls_name, final_patch_func_or_cls)
        self.is_applied = True

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
                    sys.modules[path].__file__ = 'dcu_megatron.dummy_module.py'
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
            setattr(sys.modules[module_path], function_name, None)
        return sys.modules[module_path], getattr(sys.modules[module_path], function_name) if function_name is not None else None


class MegatronPatchesManager:
    patches_info = {}

    @staticmethod
    def register_patch(orig_func_or_cls_name, new_func_or_cls=None, force_patch=False, create_dummy=False):
        if orig_func_or_cls_name not in MegatronPatchesManager.patches_info:
            MegatronPatchesManager.patches_info[orig_func_or_cls_name] = Patch(orig_func_or_cls_name, new_func_or_cls, create_dummy)
        else:
            MegatronPatchesManager.patches_info.get(orig_func_or_cls_name).set_patch_func(new_func_or_cls, force_patch)

    @staticmethod
    def apply_patches():
        for patch in MegatronPatchesManager.patches_info.values():
            patch.apply_patch()
