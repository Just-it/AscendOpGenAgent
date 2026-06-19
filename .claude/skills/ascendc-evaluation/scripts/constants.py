import torch

DEFAULT_TOLERANCES = {
    torch.bfloat16: {
        "atol": 1e-2, "rtol": 1e-2, "ulp_tol": 2,
        "sv_th": 2**-8, "sv_err": 2**-16,
        "max_re_ratio_limit": 10.0,
        "mean_re_ratio_limit": 2.0,
        "rmse_ratio_limit": 2.0,
    },
    torch.float16: {
        "atol": 1e-3, "rtol": 1e-3, "ulp_tol": 2,
        "sv_th": 2**-11, "sv_err": 2**-16,
        "max_re_ratio_limit": 10.0,
        "mean_re_ratio_limit": 2.0,
        "rmse_ratio_limit": 2.0,
    },
    torch.float32: {
        "atol": 1e-5, "rtol": 1e-5, "ulp_tol": 2,
        "sv_th": 2**-14, "sv_err": 2**-30,
        "max_re_ratio_limit": 10.0,
        "mean_re_ratio_limit": 2.0,
        "rmse_ratio_limit": 2.0,
    },
}

DEFAULT_ULP_CONFIG = {
    "method": "bitwise",
    "include_subnormal": True,
}
