# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import argparse
import sys
import warnings
from functools import wraps
from dataclasses import field, make_dataclass, MISSING

from hcu_megatron.training.arguments import add_adaptor_args, get_adaptor_args


# 动态生成的 config 子类缓存, 以其基类为 key。
# 同一种 config 只生成一次, 避免每个实例都造一个一次性的类。
_DYNAMIC_CONFIG_CLASSES = {}


def _rebuild_dynamic_config(base_cls, field_names, state):
    """unpickle 时在接收端重建动态 config 实例。

    接收端按需重新生成(或复用缓存的)动态子类, 再把属性状态灌回去, 因此不要求
    接收端事先创建过同名的动态类。base_cls 是真实模块里可导入的类, 本身能正常
    pickle。
    """
    fields = [(name, object, field(init=False)) for name in field_names]
    cls = _make_picklable_dataclass(base_cls, fields)
    obj = cls.__new__(cls)
    obj.__dict__.update(state)
    return obj


def _dynamic_config_reduce(self):
    """让动态 config 以"基类 + 字段名 + 状态"的形式序列化。

    默认的 pickle 协议只记录"模块名 + 类名", 依赖接收端存在同名类; 各 rank 的
    模型结构不同, 该假设不成立(见 _make_picklable_dataclass 的说明)。
    """
    cls = type(self)
    return (
        _rebuild_dynamic_config,
        (cls.__dynamic_base__, cls.__dynamic_fields__, self.__dict__),
    )


def _make_picklable_dataclass(base_cls, fields):
    """生成 base_cls 的动态子类, 并保证它可以被 pickle。

    直接用 make_dataclass() 造出来的类无法被 pickle: pickle 序列化实例时只记录
    "模块名 + 类名", 反序列化时靠 getattr(模块, 类名) 找回类定义。而动态类
    (1) 从未被赋值到任何模块上, (2) 不传 module= 时 __module__ 取自调用方栈帧,
    经由 ABCMeta 的 model provider 调用时会落到 'abc'。于是报错:
        Can't pickle <class 'abc.GPTModelProvider'>

    这会影响 Megatron-Bridge 导出 HF 格式权重: 它在导出 QKV 时需要跨流水线并行
    (PP) 广播模型 config, 而 broadcast_object_list() 内部使用 pickle。PP=1 时
    不发生广播, 所以只在 PP>1 时暴露。

    仅把类注册到模块上还不够: 各 rank 持有的子模块不同, 动态类的创建顺序和数量
    也不同, 若用创建序号命名, 同一个名字在不同 rank 上会指向不同的类, 甚至在
    接收端根本不存在, 于是 unpickle 报:
        Can't get attribute '_Dynamic_Qwen3VLTransformerConfig_2'

    因此这里做两件事:
      1) 用基类的模块名+类名生成确定性名字, 与创建顺序无关, 各 rank 一致;
      2) 通过 __reduce__ 让实例以"基类 + 字段名 + 状态"序列化, 接收端按需重建,
         不要求接收端预先存在该动态类。
    动态类仍是 base_cls 的子类, isinstance 判断不受影响。
    """
    cached = _DYNAMIC_CONFIG_CLASSES.get(base_cls)
    if cached is not None:
        return cached

    cls = make_dataclass(base_cls.__name__, fields=fields, bases=(base_cls,))

    holder = sys.modules[__name__]
    # same base class yields the same name regardless of rank
    unique_name = "_Dynamic_{}_{}".format(
        base_cls.__module__.replace(".", "_"), base_cls.__name__
    )
    cls.__module__ = holder.__name__
    cls.__qualname__ = unique_name
    cls.__name__ = unique_name
    # Referenced by _dynamic_config_reduce when restoring the instance from a pickled state
    cls.__dynamic_base__ = base_cls
    cls.__dynamic_fields__ = tuple(f[0] for f in fields)
    cls.__reduce__ = _dynamic_config_reduce
    setattr(holder, unique_name, cls)

    _DYNAMIC_CONFIG_CLASSES[base_cls] = cls
    return cls


def transformer_config_post_init_wrapper(post_init_func):
    @wraps(post_init_func)
    def wrapper(self):
        # remove experts from recompute_modules. Otherwise _post_init_ will raise error
        if self.recompute_modules is None:
            self.recompute_modules = set()
        self.recompute_modules = set(self.recompute_modules)
        recompute_experts = "experts" in self.recompute_modules
        recompute_router  = "router"  in self.recompute_modules
        recompute_mhc = "mhc" in self.recompute_modules
        self.recompute_modules.discard("experts")
        self.recompute_modules.discard("router")
        self.recompute_modules.discard("mhc")
        self.recompute_modules = list(self.recompute_modules)

        # set delay_wgrad_compute to avoid AssertionError(overlap_moe_expert_parallel_comm must be enabled when enabling delay_wgrad_compute)
        # set overlap_moe_expert_parallel_comm to avoid AssertionError
        need_delay_wgrad_compute_schedules = {"dualpipev", "zb_h1"}
        if (
            self.schedule_method in need_delay_wgrad_compute_schedules
            or (self.schedule_method == "vanilla" and self.delay_1f1b_cooldown_wgrad_compute)
        ):
            origin_delay_wgrad_compute = self.delay_wgrad_compute
            self.delay_wgrad_compute = False

            origin_overlap_moe_expert_parallel_comm = self.overlap_moe_expert_parallel_comm
            self.overlap_moe_expert_parallel_comm = False

        # Recompute specific transformer layers to save activation memory without enabling full recomputation
        # https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html#feature-6-recompute-selected-layers
        if self.recompute_layer_ids is not None:
            assert isinstance(
                self.recompute_layer_ids, list
            ), f"recompute_layer_ids={self.recompute_layer_ids} should be a list"
            recompute_layer_ids = list(set(self.recompute_layer_ids))
            assert len(recompute_layer_ids) > 0, "recompute layer ids is null"
            for layer_id in recompute_layer_ids:
                assert (
                    layer_id >= 0 and layer_id < self.num_layers
                ), f"recompute layer id must be between 0 and {self.num_layers - 1}"

        if self.recompute_mtp_layer_ids is not None:
            assert isinstance(
                self.recompute_mtp_layer_ids, list
            ), f"recompute_mtp_layer_ids={self.recompute_mtp_layer_ids} should be a list"
            recompute_mtp_layer_ids = list(set(self.recompute_mtp_layer_ids))
            assert len(recompute_mtp_layer_ids) > 0, "recompute layer ids is null"
            for layer_id in recompute_mtp_layer_ids:
                assert (
                    layer_id >= 0 and layer_id < self.mtp_num_layers
                ), f"recompute layer id must be between 0 and {self.mtp_num_layers - 1}"

        if (
            self.recompute_layer_ids is not None
            or self.recompute_mtp_layer_ids is not None
        ):
            if self.recompute_granularity != "full":
                raise ValueError(
                    f'When using recompute_layer_ids or recompute_mtp_layer_ids, recompute_granuarlity: {self.recompute_granularity} must be "full"'
                )

            if self.recompute_method is not None:
                raise ValueError(
                    f"When using recompute_layer_ids or recompute_mtp_layer_ids, recompute_method: {self.recompute_method} must be None."
                )

            # set recompute_granularity to avoid AssertionError (Using recompute_granularity: full so recompute_method must be "block" or "uniform")
            self.recompute_granularity = None

        post_init_func(self)
        if recompute_experts:
            self.recompute_modules.append("experts")
        if recompute_router:
            self.recompute_modules.append("router")
        if recompute_mhc:
            self.recompute_modules.append("mhc")

        if (
            self.schedule_method in need_delay_wgrad_compute_schedules
            or (self.schedule_method == "vanilla" and self.delay_1f1b_cooldown_wgrad_compute)
        ):
            self.delay_wgrad_compute = origin_delay_wgrad_compute
            self.overlap_moe_expert_parallel_comm = origin_overlap_moe_expert_parallel_comm

        # Validation for "mhc" in recompute_modules
        if self.recompute_granularity == "selective" and "mhc" in self.recompute_modules:
            if not self.enable_hyper_connections:
                raise ValueError(
                    "'mhc' in recompute_modules requires enable_hyper_connections=True."
                )
            if "mlp" in self.recompute_modules:
                raise ValueError(
                    "'mhc' and 'mlp' in recompute_modules cannot be used together. "
                    "They use different checkpoint mechanisms that may conflict."
                )
            if self.mhc_recompute_layer_num is not None and (
                isinstance(self.mhc_recompute_layer_num, bool)
                or not isinstance(self.mhc_recompute_layer_num, int)
                or self.mhc_recompute_layer_num < 1
            ):
                raise ValueError(
                    "mhc_recompute_layer_num must be a positive integer when "
                    "'mhc' is in recompute_modules."
                )
            if self.fine_grained_activation_offloading:
                raise ValueError(
                    "'mhc' in recompute_modules is incompatible with "
                    "fine_grained_activation_offloading. The mHC recompute hook fires "
                    "before the offloading backward chunk is initialized, causing "
                    "tensor_pop on a None chunk. Disable one of them."
                )

        if self.enable_hyper_connections and not (
            self.recompute_granularity == "selective" and "mhc" in self.recompute_modules
        ):
            warnings.warn(
                "HyperConnections are enabled but 'mhc' is not in "
                "recompute_modules with selective recompute. Consider adding 'mhc' to "
                "recompute_modules with selective recompute to reduce activation memory."
            )

        # Validation for hyper_connections with MTP
        if self.enable_hyper_connections and self.mtp_num_layers is not None:
            raise ValueError(
                "enable_hyper_connections is not compatible with Multi-Token Prediction (MTP). "
                "Please disable MTP (set mtp_num_layers=None) when using hyper connections."
            )

        if self.recompute_granularity == 'selective':
            if len(self.recompute_modules) > 0:
                modules_set = set(self.recompute_modules)
                if 'experts' in modules_set or 'router' in modules_set:
                    assert 'moe' not in modules_set, (
                        "'moe' cannot be used together with 'experts' or 'router' in recompute_modules. "
                        "Please choose either 'moe' or a combination of 'experts' and/or 'router'."
                    )

        if (
            self.recompute_layer_ids is not None
            or self.recompute_mtp_layer_ids is not None
        ):
            self.recompute_granularity = "full"

    return wrapper


def field_specs_from_parser(skip: set = None):
    parser = argparse.ArgumentParser(description='Adaptor Arguments', allow_abbrev=False)
    parser = add_adaptor_args(parser)

    specs = {}
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        name = action.dest
        if name in skip:
            continue

        # get the argument type
        if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            typ = bool
        elif action.nargs in ("*", "+") or isinstance(
            action, (argparse._AppendAction, argparse._AppendConstAction)
        ):
            typ = list
        else:
            typ = action.type or (type(action.default) if action.default is not None else str)

        # get the default value
        default = action.default
        if isinstance(default, (list, dict, set)):
            snapshot = type(default)(default)
            specs[name] = (typ, field(default_factory=lambda s=snapshot: type(s)(s)))
        else:
            specs[name] = (typ, default)
    return specs


def transformer_config_init_wrapper(init_func, extra_field_specs):
    """
    extra_field_specs: dict[name -> (type, default)]
      default can be a normal value, or it can be dataclasses.field(default_factory=...)
    """
    known_extra = set(extra_field_specs)

    @wraps(init_func)
    def wrapper(self, *args, **kwargs):
        # pop the new fields out of kwargs
        extras = {k: kwargs.pop(k) for k in list(kwargs) if k in known_extra}

        try:
            adaptor_args = get_adaptor_args()
        except AssertionError:
            adaptor_args = None

        # construct a dataclass with new fields
        new_fields = []
        for name, (typ, default) in extra_field_specs.items():
            if isinstance(default, type(field())):
                new_fields.append((name, typ, default))
            else:
                new_fields.append((name, typ, field(default=default)))
        self.__class__ = _make_picklable_dataclass(self.__class__, new_fields)

        # set value for extra attrs
        for name, (typ, default) in extra_field_specs.items():
            if name in extras:
                setattr(self, name, extras[name])
            elif adaptor_args is not None and hasattr(adaptor_args, name):
                setattr(self, name, getattr(adaptor_args, name))
            elif isinstance(default, type(field())):
                factory = default.default_factory
                setattr(self, name, factory() if factory is not MISSING else default.default)
            else:
                setattr(self, name, default)

        init_func(self, *args, **kwargs)

    return wrapper


# Skip existing TransformerConfig fields to avoid conflicts
from megatron.core.transformer.transformer_config import TransformerConfig as MegatronCoreTransformerConfig
from megatron.core.transformer.transformer_config import MLATransformerConfig as MegatronCoreMLATransformerConfig

existing_attrs = {f.name for f in MegatronCoreTransformerConfig.__dataclass_fields__.values()}
extra_field_specs = field_specs_from_parser(skip=existing_attrs)
transformer_config_init_func = transformer_config_init_wrapper(
    MegatronCoreTransformerConfig.__init__, extra_field_specs
)
mla_transformer_config_init_func = transformer_config_init_wrapper(
    MegatronCoreMLATransformerConfig.__init__, extra_field_specs
)
