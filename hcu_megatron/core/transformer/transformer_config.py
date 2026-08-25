# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import sys
import warnings
from functools import wraps
from dataclasses import make_dataclass, field

from hcu_megatron.training.arguments import get_adaptor_args


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
    # 确定性命名: 同一基类在任何 rank 上都得到同样的名字。
    unique_name = "_Dynamic_{}_{}".format(
        base_cls.__module__.replace(".", "_"), base_cls.__name__
    )
    cls.__module__ = holder.__name__
    cls.__qualname__ = unique_name
    cls.__name__ = unique_name
    # 供 _dynamic_config_reduce 序列化时还原用。
    cls.__dynamic_base__ = base_cls
    cls.__dynamic_fields__ = tuple(f[0] for f in fields)
    cls.__reduce__ = _dynamic_config_reduce
    setattr(holder, unique_name, cls)

    _DYNAMIC_CONFIG_CLASSES[base_cls] = cls
    return cls


def transformer_config_post_init_wrapper(post_init_func):
    @wraps(post_init_func)
    def wrapper(self):
        args = get_adaptor_args()

        # remover experts from recompute_modules. Otherwise _post_init_ will raise error
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
            args.schedule_method in need_delay_wgrad_compute_schedules
            or (args.schedule_method == "vanilla" and args.delay_1f1b_cooldown_wgrad_compute)
        ):
            origin_delay_wgrad_compute = self.delay_wgrad_compute
            self.delay_wgrad_compute = False

            origin_overlap_moe_expert_parallel_comm = self.overlap_moe_expert_parallel_comm
            self.overlap_moe_expert_parallel_comm = False

        # Recompute specific transformer layers to save activation memory without enabling full recomputation
        # https://rocm.blogs.amd.com/software-tools-optimization/primus-moe-package/README.html#feature-6-recompute-selected-layers
        if args.recompute_layer_ids is not None:
            assert isinstance(
                args.recompute_layer_ids, list
            ), f"recompute_layer_ids={args.recompute_layer_ids} should be a list"
            recompute_layer_ids = list(set(args.recompute_layer_ids))
            assert len(recompute_layer_ids) > 0, "recompute layer ids is null"
            for layer_id in recompute_layer_ids:
                assert (
                    layer_id >= 0 and layer_id < self.num_layers
                ), f"recompute layer id must be between 0 and {args.num_layers - 1}"

        if args.recompute_mtp_layer_ids is not None:
            assert isinstance(
                args.recompute_mtp_layer_ids, list
            ), f"recompute_mtp_layer_ids={args.recompute_mtp_layer_ids} should be a list"
            recompute_mtp_layer_ids = list(set(args.recompute_mtp_layer_ids))
            assert len(recompute_mtp_layer_ids) > 0, "recompute layer ids is null"
            for layer_id in recompute_mtp_layer_ids:
                assert (
                    layer_id >= 0 and layer_id < self.mtp_num_layers
                ), f"recompute layer id must be between 0 and {args.mtp_num_layers - 1}"

        if (
            args.recompute_layer_ids is not None
            or args.recompute_mtp_layer_ids is not None
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
            args.schedule_method in need_delay_wgrad_compute_schedules
            or (args.schedule_method == "vanilla" and args.delay_1f1b_cooldown_wgrad_compute)
        ):
            self.delay_wgrad_compute = origin_delay_wgrad_compute
            self.overlap_moe_expert_parallel_comm = origin_overlap_moe_expert_parallel_comm

        fields = []
        for key, value in vars(args).items():
            field_name = str(key)
            field_type = type(value)
            if not hasattr(self, key):
                field_def = (field_name, field_type, field(init=False))
                fields.append(field_def)
        # self.__class__ = make_dataclass(self.__class__.__name__, fields=fields, bases=(self.__class__,))
        # 改用可被 pickle 的动态类, 否则 PP>1 导出 HF 权重时广播 config 会失败
        self.__class__ = _make_picklable_dataclass(self.__class__, fields)

        for key, value in vars(args).items():
            if not hasattr(self, key):
                setattr(self, key, value)

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
            args.recompute_layer_ids is not None
            or args.recompute_mtp_layer_ids is not None
        ):
            self.recompute_granularity = "full"

    return wrapper
