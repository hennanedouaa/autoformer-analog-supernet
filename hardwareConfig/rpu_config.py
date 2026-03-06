# =========================================================
# RPU Configuration (EXACT copy from AnalogNAS repository)
# ---------------------------------------------------------
# This RPU configuration is taken directly from:
#
# H. Benmeziane, C. Lammie, I. Boybat, M. Rasch,
# M. L. Gallo, H. Tsai, K. E. Maghraoui, et al.
# "AnalogNAS: A Neural Network Design Framework for Accurate
# Inference with Analog In-Memory Computing."
# IEEE Edge 2023.
#
# Paper:
# https://arxiv.org/abs/2305.10459
#
# Official repository:
# https://github.com/IBM/analog-nas
#
# If you use this configuration in research, please cite
# the above paper.
# =========================================================

from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    IOParameters,
    WeightModifierType,
    WeightClipType,
    WeightRemapType,
    NoiseManagementType,
    BoundManagementType,
    WeightModifierParameter,
    WeightClipParameter,
    WeightRemapParameter,
    PrePostProcessingParameter
)
from aihwkit.inference import PCMLikeNoiseModel, GlobalDriftCompensation


#Taken from (AnalogBench,2025) repository
def gen_rpu_config():
    rpu_config = InferenceRPUConfig()
    rpu_config.modifier.std_dev = 0.06
    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL

    rpu_config.mapping.digital_bias = True
    rpu_config.mapping.weight_scaling_omega = 1.0
    rpu_config.mapping.weight_scaling_columnwise = False
    rpu_config.mapping.out_scaling_columnwise = False
    rpu_config.remap.type = WeightRemapType.LAYERWISE_SYMMETRIC

    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN
    rpu_config.clip.sigma = 2.0

    rpu_config.forward = IOParameters()
    rpu_config.forward.is_perfect = False
    rpu_config.forward.out_noise = 0.04
    rpu_config.forward.inp_bound = 1.0
    rpu_config.forward.inp_res = 1 / (2**8 - 2)
    rpu_config.forward.out_bound = 10
    rpu_config.forward.out_res = 1 / (2**8 - 2)
    rpu_config.forward.bound_management = BoundManagementType.NONE
    rpu_config.forward.noise_management = NoiseManagementType.NONE

    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.decay = 0.01
    rpu_config.pre_post.input_range.init_from_data = 50
    rpu_config.pre_post.input_range.init_std_alpha = 3.0
    rpu_config.pre_post.input_range.input_min_percentage = 0.995
    rpu_config.pre_post.input_range.manage_output_clipping = False

    rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0)
    rpu_config.drift_compensation = GlobalDriftCompensation()
    return rpu_config