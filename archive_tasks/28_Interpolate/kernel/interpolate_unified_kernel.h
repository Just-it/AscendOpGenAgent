#ifndef INTERPOLATE_UNIFIED_KERNEL_H
#define INTERPOLATE_UNIFIED_KERNEL_H

// Round 4 / Lesson 4 avoidance:
//   - Don't compute the row_mix intermediate (separable in H first).  In fp32
//     the cubic kernel weights have pattern [-,+,+,-] which causes catastrophic
//     cancellation inside row_mix; that ulp-level error is then amplified by
//     the second-stage W-axis 4-tap sum.
//   - Instead: per output (nc, h_out, w_out), directly iterate the K_h*K_w
//     pairs and Kahan-compensated-sum all 16 weighted terms in one go.
//     With 16 terms (vs separable's 4+4), Kahan actually pays off:
//     accumulated error drops to ~ eps^2 * sum(|term|) which is well below
//     the fp32 ulp at the output magnitude.

#include "kernel_operator.h"
#include "interpolate_tiling.h"

template <typename T_IN>
class InterpolateUnifiedKernel {
public:
    __aicore__ inline InterpolateUnifiedKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h_idx, GM_ADDR w_idx,
                                GM_ADDR h_w, GM_ADDR w_w, GM_ADDR y,
                                GM_ADDR tilingGm, AscendC::TPipe *pipe)
    {
        auto tilingPtr = reinterpret_cast<__gm__ InterpolateTiling *>(tilingGm);
        tiling_.NC          = tilingPtr->NC;
        tiling_.H_in        = tilingPtr->H_in;
        tiling_.W_in        = tilingPtr->W_in;
        tiling_.H_out       = tilingPtr->H_out;
        tiling_.W_out       = tilingPtr->W_out;
        tiling_.K_h         = tilingPtr->K_h;
        tiling_.K_w         = tilingPtr->K_w;
        tiling_.usedCoreNum = tilingPtr->usedCoreNum;
        tiling_.tasksPerCore= tilingPtr->tasksPerCore;
        tiling_.totalTasks  = tilingPtr->totalTasks;
        pipe_ = pipe;

        const int32_t NC    = tiling_.NC;
        const int32_t H_in  = tiling_.H_in;
        const int32_t W_in  = tiling_.W_in;
        const int32_t H_out = tiling_.H_out;
        const int32_t W_out = tiling_.W_out;
        const int32_t K_h   = tiling_.K_h;
        const int32_t K_w   = tiling_.K_w;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T_IN *>(x),
                             static_cast<uint64_t>(NC) * H_in * W_in);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T_IN *>(y),
                             static_cast<uint64_t>(NC) * H_out * W_out);
        hIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(h_idx),
                                static_cast<uint64_t>(H_out) * K_h);
        wIdxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(w_idx),
                                static_cast<uint64_t>(W_out) * K_w);
        hWGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(h_w),
                              static_cast<uint64_t>(H_out) * K_h);
        wWGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(w_w),
                              static_cast<uint64_t>(W_out) * K_w);

        W_in_pad_   = (W_in  + 31) & ~31;
        W_out_pad_  = (W_out + 31) & ~31;
        const int32_t H_out_K_h_pad = (H_out * K_h + 31) & ~31;
        const int32_t W_out_K_w_pad = (W_out * K_w + 31) & ~31;

        // K_h cached fp32 source rows (we keep them all so direct 16-tap can
        // gather without re-reading GM per output).
        pipe_->InitBuffer(xRowBuf_,
            static_cast<uint32_t>(W_in_pad_) * sizeof(T_IN));
        pipe_->InitBuffer(xRowFp32Buf_,
            static_cast<uint32_t>(INTERP_K_MAX) * W_in_pad_ * sizeof(float));
        pipe_->InitBuffer(yAccBuf_,
            static_cast<uint32_t>(W_out_pad_) * sizeof(float));
        pipe_->InitBuffer(yRowOutBuf_,
            static_cast<uint32_t>(W_out_pad_) * sizeof(T_IN));
        pipe_->InitBuffer(hIdxBuf_,
            static_cast<uint32_t>(H_out_K_h_pad) * sizeof(int32_t));
        pipe_->InitBuffer(wIdxBuf_,
            static_cast<uint32_t>(W_out_K_w_pad) * sizeof(int32_t));
        pipe_->InitBuffer(hWBuf_,
            static_cast<uint32_t>(H_out_K_h_pad) * sizeof(float));
        pipe_->InitBuffer(wWBuf_,
            static_cast<uint32_t>(W_out_K_w_pad) * sizeof(float));

        xRow_      = xRowBuf_.Get<T_IN>();
        xRowFp32_  = xRowFp32Buf_.Get<float>();
        yAcc_      = yAccBuf_.Get<float>();
        hIdxLocal_ = hIdxBuf_.Get<int32_t>();
        wIdxLocal_ = wIdxBuf_.Get<int32_t>();
        hWLocal_   = hWBuf_.Get<float>();
        wWLocal_   = wWBuf_.Get<float>();

        LoadIndexTables();
    }

    __aicore__ inline void Process()
    {
        const int32_t coreIdx = AscendC::GetBlockIdx();
        if (coreIdx >= tiling_.usedCoreNum) return;
        const int32_t taskStart = coreIdx * tiling_.tasksPerCore;
        const int32_t taskEnd =
            (taskStart + tiling_.tasksPerCore) < tiling_.totalTasks
                ? (taskStart + tiling_.tasksPerCore)
                : tiling_.totalTasks;
        for (int32_t t = taskStart; t < taskEnd; ++t) {
            const int32_t nc    = t / tiling_.H_out;
            const int32_t h_out = t - nc * tiling_.H_out;
            ProcessOne(nc, h_out);
        }
    }

private:
    __aicore__ inline void LoadIndexTables()
    {
        const int32_t H_out = tiling_.H_out;
        const int32_t W_out = tiling_.W_out;
        const int32_t K_h   = tiling_.K_h;
        const int32_t K_w   = tiling_.K_w;

        AscendC::DataCopyExtParams hIdxParams{
            1, static_cast<uint32_t>(H_out * K_h * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> hIdxPad{true, 0, 0, 0};
        AscendC::DataCopyPad(hIdxLocal_, hIdxGm_, hIdxParams, hIdxPad);

        AscendC::DataCopyExtParams wIdxParams{
            1, static_cast<uint32_t>(W_out * K_w * sizeof(int32_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> wIdxPad{true, 0, 0, 0};
        AscendC::DataCopyPad(wIdxLocal_, wIdxGm_, wIdxParams, wIdxPad);

        AscendC::DataCopyExtParams hWParams{
            1, static_cast<uint32_t>(H_out * K_h * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> hWPad{true, 0, 0, 0.0f};
        AscendC::DataCopyPad(hWLocal_, hWGm_, hWParams, hWPad);

        AscendC::DataCopyExtParams wWParams{
            1, static_cast<uint32_t>(W_out * K_w * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> wWPad{true, 0, 0, 0.0f};
        AscendC::DataCopyPad(wWLocal_, wWGm_, wWParams, wWPad);

        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline float CubicKernelNpuFp32(float t_signed) {
        // Mirrors PyTorch's cubic_convolution1/2 with A=-0.75, but every
        // intermediate stays in NPU fp32 — same rounding pipeline as the
        // PyTorch NPU reference path.
        const float A = -0.75f;
        float t = t_signed >= 0.0f ? t_signed : -t_signed;
        if (t <= 1.0f) {
            // ((A+2)*t - (A+3)) * t * t + 1
            float ap2 = A + 2.0f;
            float ap3 = A + 3.0f;
            float s1  = ap2 * t - ap3;
            float s2  = s1 * t;
            float s3  = s2 * t;
            return s3 + 1.0f;
        }
        if (t < 2.0f) {
            // ((A*t - 5*A) * t + 8*A) * t - 4*A
            float ax  = A * t;
            float m5a = 5.0f * A;
            float sub = ax - m5a;
            float s1  = sub * t;
            float a8  = 8.0f * A;
            float s2  = s1 + a8;
            float s3  = s2 * t;
            float a4  = 4.0f * A;
            return s3 - a4;
        }
        return 0.0f;
    }

    __aicore__ inline void ProcessOne(int32_t nc, int32_t h_out)
    {
        const int32_t H_in  = tiling_.H_in;
        const int32_t W_in  = tiling_.W_in;
        const int32_t W_out = tiling_.W_out;
        const int32_t K_h   = tiling_.K_h;
        const int32_t K_w   = tiling_.K_w;
        const bool bic_kern = (tiling_.bicubic_in_kernel != 0);

        // For bicubic-in-kernel, reuse the first slot of h_w / w_w to pass t.
        // Indices h_idx still come from host (clamping requires per-element
        // boundary knowledge that's cleaner to do once on host).
        // For other modes, h_w / w_w are full K_h / K_w precomputed weights.

        // ---- Load K_h source rows (X[nc, h_idx[h_out, kh], :]) into UB.
        for (int32_t kh = 0; kh < K_h; ++kh) {
            int32_t hi = hIdxLocal_.GetValue(h_out * K_h + kh);
            if (hi < 0) hi = 0;
            if (hi > H_in - 1) hi = H_in - 1;
            uint64_t srcOffset = (static_cast<uint64_t>(nc) * H_in + hi) * W_in;
            AscendC::DataCopyExtParams cp{
                1, static_cast<uint32_t>(W_in * sizeof(T_IN)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T_IN> cpPad{
                true, 0, 0, static_cast<T_IN>(0)};
            AscendC::DataCopyPad(xRow_, xGm_[srcOffset], cp, cpPad);
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::LocalTensor<float> dstK = xRowFp32_[kh * W_in_pad_];
            if constexpr (std::is_same_v<T_IN, float>) {
                AscendC::DataCopy(dstK, xRow_, W_in_pad_);
            } else {
                AscendC::Cast(dstK, xRow_, AscendC::RoundMode::CAST_NONE,
                              W_in_pad_);
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ---- Direct 16-tap weighted sum with Kahan compensation per output.
        // Reverted to host-side precomputed weights (Round 5 best result was
        // 71/73 with host fp32-step-by-step weights; kernel-side bicubic
        // broke other cases catastrophically).
        for (int32_t w_out = 0; w_out < W_out; ++w_out) {
            // Best-known ordering (Round 7): kh-outer kw-inner, mul order
            // (input * wh) * ww + 16-tap Kahan compensated sum.
            float acc  = 0.0f;
            float comp = 0.0f;
            for (int32_t kh = 0; kh < K_h; ++kh) {
                const float wh = hWLocal_.GetValue(h_out * K_h + kh);
                if (wh == 0.0f) continue;
                AscendC::LocalTensor<float> rowK = xRowFp32_[kh * W_in_pad_];
                for (int32_t kw = 0; kw < K_w; ++kw) {
                    const float ww = wWLocal_.GetValue(w_out * K_w + kw);
                    if (ww == 0.0f) continue;
                    int32_t wi = wIdxLocal_.GetValue(w_out * K_w + kw);
                    if (wi < 0) wi = 0;
                    if (wi > W_in - 1) wi = W_in - 1;
                    const float input_val = rowK.GetValue(wi);
                    const float partial   = input_val * wh;
                    const float term      = partial * ww;
                    const float y = term - comp;
                    const float t = acc + y;
                    comp = (t - acc) - y;
                    acc = t;
                }
            }
            yAcc_.SetValue(w_out, acc);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ---- Cast & store.
        AscendC::LocalTensor<T_IN> yOut = yRowOutBuf_.Get<T_IN>();
        if constexpr (std::is_same_v<T_IN, float>) {
            AscendC::DataCopy(yOut, yAcc_, W_out_pad_);
        } else if constexpr (std::is_same_v<T_IN, bfloat16_t>) {
            AscendC::Cast(yOut, yAcc_, AscendC::RoundMode::CAST_ROUND,
                          W_out_pad_);
        } else {
            AscendC::Cast(yOut, yAcc_, AscendC::RoundMode::CAST_NONE,
                          W_out_pad_);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        uint64_t dstOffset =
            (static_cast<uint64_t>(nc) * tiling_.H_out + h_out) * tiling_.W_out;
        AscendC::DataCopyExtParams outParams{
            1, static_cast<uint32_t>(W_out * sizeof(T_IN)), 0, 0, 0};
        AscendC::DataCopyPad(yGm_[dstOffset], yOut, outParams);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    InterpolateTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};

    AscendC::GlobalTensor<T_IN>    xGm_;
    AscendC::GlobalTensor<T_IN>    yGm_;
    AscendC::GlobalTensor<int32_t> hIdxGm_;
    AscendC::GlobalTensor<int32_t> wIdxGm_;
    AscendC::GlobalTensor<float>   hWGm_;
    AscendC::GlobalTensor<float>   wWGm_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xRowBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xRowFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yAccBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yRowOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hIdxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wIdxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hWBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wWBuf_;

    AscendC::LocalTensor<T_IN>    xRow_;
    AscendC::LocalTensor<float>   xRowFp32_;
    AscendC::LocalTensor<float>   yAcc_;
    AscendC::LocalTensor<int32_t> hIdxLocal_;
    AscendC::LocalTensor<int32_t> wIdxLocal_;
    AscendC::LocalTensor<float>   hWLocal_;
    AscendC::LocalTensor<float>   wWLocal_;

    int32_t W_in_pad_{0};
    int32_t W_out_pad_{0};
};

#endif  // INTERPOLATE_UNIFIED_KERNEL_H
