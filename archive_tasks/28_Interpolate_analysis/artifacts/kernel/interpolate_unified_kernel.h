#ifndef INTERPOLATE_UNIFIED_KERNEL_H
#define INTERPOLATE_UNIFIED_KERNEL_H

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

        // Pad to 32 elements for safe vector ops.
        W_in_pad_   = (W_in  + 31) & ~31;
        W_out_pad_  = (W_out + 31) & ~31;
        const int32_t H_out_K_h_pad = (H_out * K_h + 31) & ~31;
        const int32_t W_out_K_w_pad = (W_out * K_w + 31) & ~31;

        pipe_->InitBuffer(xRowBuf_,
            static_cast<uint32_t>(W_in_pad_) * sizeof(T_IN));
        pipe_->InitBuffer(rowFp32Buf_,
            static_cast<uint32_t>(W_in_pad_) * sizeof(float));
        pipe_->InitBuffer(rowMixBuf_,
            static_cast<uint32_t>(W_in_pad_) * sizeof(float));
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
        rowFp32_   = rowFp32Buf_.Get<float>();
        rowMix_    = rowMixBuf_.Get<float>();
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

    __aicore__ inline void ProcessOne(int32_t nc, int32_t h_out)
    {
        const int32_t H_in  = tiling_.H_in;
        const int32_t W_in  = tiling_.W_in;
        const int32_t W_out = tiling_.W_out;
        const int32_t K_h   = tiling_.K_h;
        const int32_t K_w   = tiling_.K_w;

        // ---- Phase 1: rowMix = sum_kh(h_w[h_out,kh] * X[nc, h_idx[h_out,kh], :])
        AscendC::Duplicate(rowMix_, 0.0f, W_in_pad_);
        AscendC::PipeBarrier<PIPE_V>();

        for (int32_t kh = 0; kh < K_h; ++kh) {
            int32_t hi = hIdxLocal_.GetValue(h_out * K_h + kh);
            if (hi < 0) hi = 0;
            if (hi > H_in - 1) hi = H_in - 1;
            const float wh = hWLocal_.GetValue(h_out * K_h + kh);

            uint64_t srcOffset = (static_cast<uint64_t>(nc) * H_in + hi) * W_in;
            AscendC::DataCopyExtParams cp{
                1, static_cast<uint32_t>(W_in * sizeof(T_IN)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<T_IN> cpPad{
                true, 0, 0, static_cast<T_IN>(0)};
            AscendC::DataCopyPad(xRow_, xGm_[srcOffset], cp, cpPad);
            AscendC::PipeBarrier<PIPE_ALL>();

            // Cast / copy to fp32.
            if constexpr (std::is_same_v<T_IN, float>) {
                AscendC::DataCopy(rowFp32_, xRow_, W_in_pad_);
            } else {
                AscendC::Cast(rowFp32_, xRow_, AscendC::RoundMode::CAST_NONE,
                              W_in_pad_);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (wh != 0.0f) {
                // Fused multiply-add: rowMix += wh * rowFp32 (one rounding step).
                AscendC::Axpy(rowMix_, rowFp32_, wh, W_in_pad_);
                AscendC::PipeBarrier<PIPE_V>();
            }
        }

        // ---- Phase 2: Y[w_out] = sum_kw(w_w[w_out,kw] * rowMix[w_idx[w_out,kw]])
        AscendC::Duplicate(yAcc_, 0.0f, W_out_pad_);
        AscendC::PipeBarrier<PIPE_ALL>();

        // Pairwise tree reduction over K_w-tap (avoids Lesson #1, #2:
        // sequential / Kahan didn't match PyTorch NPU bicubic; try fixed
        // pairwise order ((t0+t1)+(t2+t3)) at K_w<=4).
        for (int32_t w_out = 0; w_out < W_out; ++w_out) {
            float t[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int32_t kw = 0; kw < K_w; ++kw) {
                const float ww = wWLocal_.GetValue(w_out * K_w + kw);
                if (ww == 0.0f) continue;
                int32_t wi = wIdxLocal_.GetValue(w_out * K_w + kw);
                if (wi < 0) wi = 0;
                if (wi > W_in - 1) wi = W_in - 1;
                t[kw] = ww * rowMix_.GetValue(wi);
            }
            // Pairwise: ((t0+t1) + (t2+t3))
            const float p01 = t[0] + t[1];
            const float p23 = t[2] + t[3];
            yAcc_.SetValue(w_out, p01 + p23);
        }
        AscendC::PipeBarrier<PIPE_ALL>();

        // ---- Phase 3: Cast & store ----
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
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowFp32Buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowMixBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yAccBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yRowOutBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hIdxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wIdxBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> hWBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> wWBuf_;

    AscendC::LocalTensor<T_IN>    xRow_;
    AscendC::LocalTensor<float>   rowFp32_;
    AscendC::LocalTensor<float>   rowMix_;
    AscendC::LocalTensor<float>   yAcc_;
    AscendC::LocalTensor<int32_t> hIdxLocal_;
    AscendC::LocalTensor<int32_t> wIdxLocal_;
    AscendC::LocalTensor<float>   hWLocal_;
    AscendC::LocalTensor<float>   wWLocal_;

    int32_t W_in_pad_{0};
    int32_t W_out_pad_{0};
};

#endif  // INTERPOLATE_UNIFIED_KERNEL_H
