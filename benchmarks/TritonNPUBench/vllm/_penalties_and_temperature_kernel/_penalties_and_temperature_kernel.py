# Extracted Triton Kernels
# Test file: /home/z00841464/benchmark/Triton_Automation-br_a3_operator/ascend/test/vllm_operator_cases/newtest_cases/test__penalties_and_temperature_kernel.py
# Main kernel: _penalties_and_temperature_kernel
# PT file: penalties_and_temperature_kernel_test_data_v2.pt

import triton
import triton.language as tl


# === _penalties_and_temperature_kernel ===
@triton.jit
def _penalties_and_temperature_kernel(
    logits_ptr,
    logits_stride,
    idx_mapping_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    temperature_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)
    temperature = tl.load(temperature_ptr + req_state_idx)
    temperature = tl.where(temperature == 0.0, 1.0, temperature)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    use_penalty = use_rep_penalty or (use_freq_penalty or use_pres_penalty)
    use_temperature = temperature != 1.0
    if not (use_penalty or use_temperature):
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + batch_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    if use_penalty:
        output_bin_counts = tl.load(
            output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
            mask=mask,
        )
        output_bin_mask = output_bin_counts > 0

        # Apply repetition penalties.
        if use_rep_penalty:
            packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
            packed_mask = tl.load(
                prompt_bin_mask_ptr
                + req_state_idx * prompt_bin_mask_stride
                + packed_block,
                mask=packed_block < tl.cdiv(vocab_size, 32),
            )
            prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
            prompt_bin_mask = prompt_bin_mask.to(tl.int1)
            prompt_bin_mask = prompt_bin_mask.reshape(BLOCK_SIZE)

            # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
            scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
            # If logits are positive, divide by penalty, otherwise multiply by penalty.
            logits *= tl.where(logits > 0, 1.0 / scale, scale)

        # Apply frequency penalties.
        logits -= freq_penalty * output_bin_counts
        # Apply presence penalties.
        logits -= pres_penalty * output_bin_mask

    # Apply temperature.
    logits = logits / temperature

    # Store back to logits.
    tl.store(logits_ptr + batch_idx * logits_stride + block, logits, mask=mask)

