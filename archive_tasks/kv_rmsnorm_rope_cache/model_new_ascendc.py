import torch
import torch.nn as nn
import _kv_rmsnorm_rope_cache_ext


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def postprocess_output(self, output, inputs):
        if len(inputs) >= 14:
            cache_mode = inputs[12]
            is_output_kv = inputs[13]
            if cache_mode == 'Norm' or not is_output_kv:
                return output[:2]
        return output

    def forward(self, kv, gamma, cos, sin, index, k_cache, ckv_cache,
                k_rope_scale=None, c_kv_scale=None,
                k_rope_offset=None, c_kv_offset=None,
                epsilon=1e-5, cache_mode='Norm', is_output_kv=False):
        return _kv_rmsnorm_rope_cache_ext.run_kv_rmsnorm_rope_cache(
            kv, gamma, cos, sin, index, k_cache, ckv_cache,
            epsilon, cache_mode, is_output_kv)
