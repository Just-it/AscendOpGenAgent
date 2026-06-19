"""ULP (Unit in the Last Place) distance utilities for precision testing.

Two computation modes:
  - bitwise: integer distance between IEEE-754 bit representations.
  - native:  |a - b| / ULP(b, dtype), all arithmetic in CPU fp64.

include_subnormal controls subnormal handling:
  - True:  subnormals participate; ULP(0) = min_subnormal.
  - False: subnormals flushed to 0; ULP(0) = min_normal.
  ±0 always treated as identical (distance = 0) in both modes.

Supports bf16, fp16, and fp32.
"""

import numpy as np
import torch


# ---------------------------------------------------------------------------
# IEEE-754 dtype parameters
# ---------------------------------------------------------------------------

_DTYPE_PARAMS = {
    torch.bfloat16: {"mantissa_bits": 7,  "bias": 127, "exp_mask": 0xFF},
    torch.float16:  {"mantissa_bits": 10, "bias": 15,  "exp_mask": 0x1F},
    torch.float32:  {"mantissa_bits": 23, "bias": 127, "exp_mask": 0xFF},
}


# ---------------------------------------------------------------------------
# Bit-level helpers
# ---------------------------------------------------------------------------

def _to_bf16_bits(t: torch.Tensor) -> np.ndarray:
    """Convert tensor to bf16 uint16 bit representation."""
    t32 = t.to(torch.bfloat16).to(torch.float32)
    bits32 = t32.view(torch.uint32).numpy()
    return ((bits32 >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.uint16)


def _to_fp16_bits(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.float16).view(torch.uint16).numpy()


def _to_fp32_bits(t: torch.Tensor) -> np.ndarray:
    return t.to(torch.float32).view(torch.uint32).numpy()


def _flush_bits(bits: np.ndarray, bits_width: int,
                mantissa_bits: int) -> np.ndarray:
    """Flush -0 to +0.  If include_subnormal is handled externally,
    this only fixes the ±0 issue."""
    if bits_width == 16:
        sign_bit = np.uint16(1 << 15)
        # -0 → +0
        bits = np.where(bits == sign_bit, np.uint16(0), bits)
    else:
        sign_bit = np.uint32(1 << 31)
        bits = np.where(bits == sign_bit, np.uint32(0), bits)
    return bits


def _flush_subnormals(bits: np.ndarray, bits_width: int,
                      mantissa_bits: int) -> np.ndarray:
    """Flush subnormals and ±0 to +0."""
    if bits_width == 16:
        sign_bit = np.uint16(1 << 15)
        exp_mask_shifted = np.uint16(((1 << (bits_width - 1 - mantissa_bits)) - 1) << mantissa_bits)
        exp_bits = bits & exp_mask_shifted
        # Subnormal: exp == 0 (includes ±0)
        is_subnormal_or_zero = (exp_bits == np.uint16(0))
        # Also catch -0 explicitly
        is_neg_zero = (bits == sign_bit)
        bits = np.where(is_subnormal_or_zero | is_neg_zero, np.uint16(0), bits)
    else:
        sign_bit = np.uint32(1 << 31)
        exp_mask_shifted = np.uint32(((1 << (bits_width - 1 - mantissa_bits)) - 1) << mantissa_bits)
        exp_bits = bits & exp_mask_shifted
        is_subnormal_or_zero = (exp_bits == np.uint32(0))
        is_neg_zero = (bits == sign_bit)
        bits = np.where(is_subnormal_or_zero | is_neg_zero, np.uint32(0), bits)
    return bits


def _ordered_from_uint(u: np.ndarray, bits: int) -> np.ndarray:
    """Map IEEE-754 float bits to monotonic ordered ints for ULP distance.

    Positive floats map to themselves; negative floats are mirrored
    so that the ordering is consistent across the number line.
    """
    if bits == 16:
        sign_bit = np.uint16(1 << 15)
        max_val = np.uint16(0xFFFF)
        signed = u.astype(np.int32)
        mask = (u & sign_bit).astype(bool)
        return np.where(mask, (max_val.astype(np.int32) - signed), signed)
    elif bits == 32:
        sign_bit = np.uint32(1 << 31)
        max_val = np.uint32(0xFFFFFFFF)
        signed = u.astype(np.int64)
        mask = (u & sign_bit).astype(bool)
        return np.where(mask, (max_val.astype(np.int64) - signed), signed)
    else:
        raise ValueError(f"Unsupported bit width: {bits}")


# ---------------------------------------------------------------------------
# Bitwise mode
# ---------------------------------------------------------------------------

def ulp_distance_bitwise(a: torch.Tensor, b: torch.Tensor,
                         dtype: torch.dtype,
                         include_subnormal: bool = True) -> np.ndarray:
    """Compute bitwise ULP distance between two tensors.

    When include_subnormal=False, subnormals and ±0 are flushed to +0.
    ±0 are always treated as identical (distance = 0).
    """
    a_flat = a.detach().cpu().flatten()
    b_flat = b.detach().cpu().flatten()

    params = _DTYPE_PARAMS[dtype]
    mb = params["mantissa_bits"]

    if dtype == torch.bfloat16:
        a_bits = _to_bf16_bits(a_flat)
        b_bits = _to_bf16_bits(b_flat)
        bw = 16
    elif dtype == torch.float16:
        a_bits = _to_fp16_bits(a_flat)
        b_bits = _to_fp16_bits(b_flat)
        bw = 16
    elif dtype == torch.float32:
        a_bits = _to_fp32_bits(a_flat)
        b_bits = _to_fp32_bits(b_flat)
        bw = 32
    else:
        raise ValueError(f"Unsupported dtype for ULP: {dtype}")

    if include_subnormal:
        # Only flush -0 → +0
        a_bits = _flush_bits(a_bits, bw, mb)
        b_bits = _flush_bits(b_bits, bw, mb)
    else:
        # Flush subnormals and ±0 → +0
        a_bits = _flush_subnormals(a_bits, bw, mb)
        b_bits = _flush_subnormals(b_bits, bw, mb)

    a_ord = _ordered_from_uint(a_bits, bw)
    b_ord = _ordered_from_uint(b_bits, bw)

    return np.abs(a_ord - b_ord)


# ---------------------------------------------------------------------------
# Native mode: |a - b| / ULP(b, dtype)
# ---------------------------------------------------------------------------

def _ulp_value_at(b: torch.Tensor, dtype: torch.dtype,
                  include_subnormal: bool = True) -> np.ndarray:
    """Compute the ULP (spacing) at each element of b in the given dtype.

    Returns fp64 numpy array.  inf/nan positions yield nan.

    include_subnormal=True:
        Normal:  ULP = 2^(E - bias - mantissa_bits)
        Sub/zero: ULP = 2^(1 - bias - mantissa_bits)  (min_subnormal spacing)
    include_subnormal=False:
        Normal:  ULP = 2^(E - bias - mantissa_bits)
        Zero:    ULP = 2^(1 - bias)  (min_normal, the next representable after 0)
    """
    params = _DTYPE_PARAMS.get(dtype)
    if params is None:
        raise ValueError(f"Unsupported dtype for ULP: {dtype}")

    mantissa_bits = params["mantissa_bits"]
    bias = params["bias"]
    exp_mask = params["exp_mask"]

    b_flat = b.detach().cpu().flatten()

    if dtype == torch.bfloat16:
        b32 = b_flat.to(torch.bfloat16).to(torch.float32)
        bits32 = b32.view(torch.uint32).numpy()
        bf16_bits = ((bits32 >> np.uint32(16)) & np.uint32(0xFFFF)).astype(np.uint16)
        exp = ((bf16_bits >> np.uint16(mantissa_bits)) & np.uint16(exp_mask)).astype(np.int32)
    elif dtype == torch.float16:
        bits = b_flat.to(torch.float16).view(torch.uint16).numpy()
        exp = ((bits >> np.uint16(mantissa_bits)) & np.uint16(exp_mask)).astype(np.int32)
    else:  # fp32
        bits = b_flat.to(torch.float32).view(torch.uint32).numpy()
        exp = ((bits >> np.uint32(mantissa_bits)) & np.uint32(exp_mask)).astype(np.int32)

    if include_subnormal:
        # Subnormals share E=1 spacing: ULP = 2^(1 - bias - mantissa_bits)
        eff_exp = np.maximum(exp, 1)
        ulp_values = np.ldexp(1.0, (eff_exp - bias - mantissa_bits).astype(np.int64))
    else:
        # Normal: standard formula.  Zero/subnormal: ULP = min_normal = 2^(1-bias)
        ulp_values = np.where(
            exp > 0,
            np.ldexp(1.0, (exp - bias - mantissa_bits).astype(np.int64)),
            np.ldexp(1.0, np.int64(1 - bias)),
        )

    # inf/nan (all-ones exponent) → nan
    ulp_values[exp == exp_mask] = np.nan

    return ulp_values


def ulp_distance_native(a: torch.Tensor, b: torch.Tensor,
                        dtype: torch.dtype,
                        include_subnormal: bool = True) -> np.ndarray:
    """Compute native ULP distance: |a - b| / ULP(b, dtype).

    b is cast to *dtype* to determine the ULP at each position.
    All error arithmetic in CPU fp64.  inf/nan positions yield nan.

    When include_subnormal=False, subnormal values in a and b are
    flushed to 0 before computing |a - b|.
    """
    a64 = a.detach().cpu().flatten().double().numpy()
    b64 = b.detach().cpu().flatten().double().numpy()

    if not include_subnormal:
        params = _DTYPE_PARAMS[dtype]
        # min_normal = 2^(1-bias)
        min_normal = np.ldexp(1.0, 1 - params["bias"])
        a64 = np.where(np.abs(a64) < min_normal, 0.0, a64)
        b64 = np.where(np.abs(b64) < min_normal, 0.0, b64)

    ulp_at_b = _ulp_value_at(b, dtype, include_subnormal=include_subnormal)

    with np.errstate(divide='ignore', invalid='ignore'):
        distance = np.abs(a64 - b64) / ulp_at_b

    return distance


# ---------------------------------------------------------------------------
# Unified metrics
# ---------------------------------------------------------------------------

def ulp_metrics(a: torch.Tensor, b: torch.Tensor,
                dtype: torch.dtype, ulp_tol: int = 2,
                method: str = "bitwise",
                include_subnormal: bool = True) -> dict:
    """Compute ULP metrics: ulp_mean, ulp_max, ulp_miss_rate.

    method='bitwise': integer distance between IEEE-754 bit representations.
    method='native':  |a - b| / ULP(b, dtype), computed in fp64.
    include_subnormal: if False, flush subnormals to 0 before computing.
    """
    if method == "native":
        dist = ulp_distance_native(a, b, dtype,
                                   include_subnormal=include_subnormal)
        # Filter out nan/inf (from inf/nan positions in the input)
        valid = np.isfinite(dist)
        dist = dist[valid]
    else:
        dist = ulp_distance_bitwise(a, b, dtype,
                                    include_subnormal=include_subnormal)

    n = dist.size
    if n == 0:
        return {"ulp_mean": 0.0, "ulp_max": 0.0, "ulp_miss_rate": 0.0}
    return {
        "ulp_mean": float(np.mean(dist)),
        "ulp_max": int(np.max(dist)) if method == "bitwise" else float(np.max(dist)),
        "ulp_miss_rate": float(np.sum(dist > ulp_tol) / n),
    }
