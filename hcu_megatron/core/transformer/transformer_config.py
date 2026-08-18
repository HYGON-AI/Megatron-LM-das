# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
import warnings
from functools import wraps
from dataclasses import make_dataclass, field

from hcu_megatron.training.arguments import get_adaptor_args


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
        self.__class__ = make_dataclass(self.__class__.__name__, fields=fields, bases=(self.__class__,))

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
