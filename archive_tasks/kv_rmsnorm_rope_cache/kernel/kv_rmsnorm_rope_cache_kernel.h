#pragma once

#include "kernel_operator.h"
#include "kernel_common.h"
#include "kv_rmsnorm_rope_cache_tiling.h"

template <typename dataType>
class KvRmsnormRopeCacheKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin,
        GM_ADDR index, GM_ADDR k_cache, GM_ADDR ckv_cache,
        GM_ADDR k_cache_out, GM_ADDR ckv_cache_out,
        GM_ADDR k_embed_out, GM_ADDR v_out,
        GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);

        int32_t hidden_size = tiling_.hidden_size;
        int32_t total = tiling_.total;

        kvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(kv), total * hidden_size);
        gammaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gamma), tiling_.rms_size);
        cosGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(cos), total * tiling_.rope_size);
        sinGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(sin), total * tiling_.rope_size);
        indexGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(index), tiling_.index_numel);
        kCacheGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(k_cache),
            tiling_.k_cache_dim0 * tiling_.k_cache_dim1 * tiling_.k_cache_dim2 * tiling_.k_cache_dim3);
        ckvCacheGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(ckv_cache),
            tiling_.ckv_cache_dim0 * tiling_.ckv_cache_dim1 * tiling_.ckv_cache_dim2 * tiling_.ckv_cache_dim3);
        kCacheOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(k_cache_out),
            tiling_.k_cache_dim0 * tiling_.k_cache_dim1 * tiling_.k_cache_dim2 * tiling_.k_cache_dim3);
        ckvCacheOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(ckv_cache_out),
            tiling_.ckv_cache_dim0 * tiling_.ckv_cache_dim1 * tiling_.ckv_cache_dim2 * tiling_.ckv_cache_dim3);

        if (tiling_.is_output_kv) {
            kEmbedOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(k_embed_out), total * tiling_.rope_size);
            vOutGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(v_out), total * tiling_.rms_size);
        }

        if ASCEND_IS_AIV {
            pipe_ = pipe;

            pipe_->InitBuffer(gammaInQueue_, 1, tiling_.rms_size * sizeof(dataType));
            pipe_->InitBuffer(rmsInQueue_, 1, tiling_.rms_size * sizeof(dataType));
            pipe_->InitBuffer(ropeInQueue_, 1, tiling_.rope_size * sizeof(dataType));
            pipe_->InitBuffer(cosInQueue_, 1, tiling_.rope_size * sizeof(dataType));
            pipe_->InitBuffer(sinInQueue_, 1, tiling_.rope_size * sizeof(dataType));
            pipe_->InitBuffer(vOutQueue_, 1, tiling_.rms_size * sizeof(dataType));
            pipe_->InitBuffer(kEmbedOutQueue_, 1, tiling_.rope_size * sizeof(dataType));

            pipe_->InitBuffer(reduceBuf_, tiling_.rms_size * sizeof(float));
            pipe_->InitBuffer(sumBuf_, 32 * sizeof(float));
            pipe_->InitBuffer(invRmsBuf_, 32 * sizeof(float));
            pipe_->InitBuffer(vFloatBuf_, tiling_.rms_size * sizeof(float));
            pipe_->InitBuffer(ropeInFloatBuf_, tiling_.rope_size * sizeof(float));
            pipe_->InitBuffer(rotateHalfFloatBuf_, tiling_.rope_size * sizeof(float));
            pipe_->InitBuffer(kEmbedFloatBuf_, tiling_.rope_size * sizeof(float));
            pipe_->InitBuffer(scatterKBuf_, 32 * sizeof(dataType));
            pipe_->InitBuffer(scatterVBuf_, 32 * sizeof(dataType));

            if constexpr (!std::is_same_v<dataType, float>) {
                pipe_->InitBuffer(rmsCastBuf_, tiling_.rms_size * sizeof(float));
                pipe_->InitBuffer(ropeCastBuf_, tiling_.rope_size * sizeof(float));
                pipe_->InitBuffer(cosCastBuf_, tiling_.rope_size * sizeof(float));
                pipe_->InitBuffer(sinCastBuf_, tiling_.rope_size * sizeof(float));
                pipe_->InitBuffer(vCastBuf_, tiling_.rms_size * sizeof(float));
                pipe_->InitBuffer(kEmbedCastBuf_, tiling_.rope_size * sizeof(float));
                pipe_->InitBuffer(gammaCastBuf_, tiling_.rms_size * sizeof(float));
            }

            AscendC::LocalTensor<dataType> gammaInTmp;
            gammaInQueue_.AllocTensor<dataType>(gammaInTmp);
            LoadGmToUb(gammaInTmp, gammaGM_, static_cast<uint32_t>(tiling_.rms_size));
            gammaInQueue_.EnQue(gammaInTmp);
            gammaInQueue_.DeQue<dataType>(gammaInTmp);
            AscendC::PipeBarrier<PIPE_MTE2>();

            gammaFloatLocal_ = gammaCastBuf_.Get<float>();
            AscendC::Cast(gammaFloatLocal_, gammaInTmp, AscendC::RoundMode::CAST_NONE, tiling_.rms_size);
            AscendC::PipeBarrier<PIPE_V>();
            gammaInQueue_.FreeTensor(gammaInTmp);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int blockIdx = AscendC::GetBlockIdx();

            for (int localIdx = 0; localIdx < tiling_.tasksPerCore; ++localIdx) {
                const int bx = blockIdx * tiling_.tasksPerCore + localIdx;
                if (bx >= BlockCount()) {
                    continue;
                }
                for (int row = 0; row < tiling_.block_size; ++row) {
                    const int pos = bx * tiling_.block_size + row;
                    if (pos < tiling_.total) {
                        ProcessPosition(pos);
                    }
                }
            }
        }
    }

private:
    __aicore__ inline int32_t BlockCount() const
    {
        return (tiling_.total + tiling_.block_size - 1) / tiling_.block_size;
    }

    __aicore__ inline AscendC::RoundMode OutputRoundMode() const
    {
        if constexpr (std::is_same_v<dataType, bfloat16_t>) {
            return AscendC::RoundMode::CAST_ROUND;
        }
        return AscendC::RoundMode::CAST_NONE;
    }

    __aicore__ inline void LoadGmToUb(AscendC::LocalTensor<dataType> dst,
                                      AscendC::GlobalTensor<dataType> src, uint32_t count)
    {
        AscendC::DataCopy(dst, src, count);
    }

    __aicore__ inline void StoreUbToGm(AscendC::GlobalTensor<dataType> dst,
                                       AscendC::LocalTensor<dataType> src, uint32_t count)
    {
        AscendC::DataCopy(dst, src, count);
    }

    __aicore__ inline void PrepareInputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &src,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = src.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
            AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void PrepareOutputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &out,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            (void)out;
            dst = castBuf.Get<float>();
        }
    }

    __aicore__ inline void FinalizeOutputTensor(
        AscendC::LocalTensor<dataType> &out,
        AscendC::LocalTensor<float> &src,
        int32_t count)
    {
        if constexpr (!std::is_same_v<dataType, float>) {
            AscendC::Cast(out, src, OutputRoundMode(), count);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyInInputs(int32_t pos)
    {
        rmsInQueue_.AllocTensor<dataType>(rmsInLocal_);
        LoadGmToUb(rmsInLocal_, kvGM_[pos * tiling_.hidden_size], static_cast<uint32_t>(tiling_.rms_size));
        rmsInQueue_.EnQue(rmsInLocal_);

        ropeInQueue_.AllocTensor<dataType>(ropeInLocal_);
        LoadGmToUb(ropeInLocal_, kvGM_[pos * tiling_.hidden_size + tiling_.rms_size], static_cast<uint32_t>(tiling_.rope_size));
        ropeInQueue_.EnQue(ropeInLocal_);

        cosInQueue_.AllocTensor<dataType>(cosInLocal_);
        LoadGmToUb(cosInLocal_, cosGM_[pos * tiling_.rope_size], static_cast<uint32_t>(tiling_.rope_size));
        cosInQueue_.EnQue(cosInLocal_);

        sinInQueue_.AllocTensor<dataType>(sinInLocal_);
        LoadGmToUb(sinInLocal_, sinGM_[pos * tiling_.rope_size], static_cast<uint32_t>(tiling_.rope_size));
        sinInQueue_.EnQue(sinInLocal_);
    }

    __aicore__ inline void CopyOutOutputs(int32_t pos, int32_t b, int32_t s, int32_t n)
    {
        vOutQueue_.DeQue<dataType>(vOutLocal_);
        kEmbedOutQueue_.DeQue<dataType>(kEmbedOutLocal_);

        UpdateCache(b, s, n, vOutLocal_, kEmbedOutLocal_);

        if (tiling_.is_output_kv) {
            StoreUbToGm(vOutGM_[pos * tiling_.rms_size], vOutLocal_, static_cast<uint32_t>(tiling_.rms_size));
            StoreUbToGm(kEmbedOutGM_[pos * tiling_.rope_size], kEmbedOutLocal_, static_cast<uint32_t>(tiling_.rope_size));
        }

        vOutQueue_.FreeTensor(vOutLocal_);
        kEmbedOutQueue_.FreeTensor(kEmbedOutLocal_);
    }

    __aicore__ inline void ProcessPosition(int32_t pos)
    {
        int32_t rms_size = tiling_.rms_size;
        int32_t rope_size = tiling_.rope_size;

        CopyInInputs(pos);

        // DeQue inputs
        rmsInQueue_.DeQue<dataType>(rmsInLocal_);
        ropeInQueue_.DeQue<dataType>(ropeInLocal_);
        cosInQueue_.DeQue<dataType>(cosInLocal_);
        sinInQueue_.DeQue<dataType>(sinInLocal_);

        // Cast inputs to float
        AscendC::LocalTensor<float> rmsInFloat;
        PrepareInputTensor(rmsInFloat, rmsInLocal_, rmsCastBuf_, rms_size);
        AscendC::LocalTensor<float> ropeInFloat;
        PrepareInputTensor(ropeInFloat, ropeInLocal_, ropeCastBuf_, rope_size);
        AscendC::LocalTensor<float> cosFloat;
        PrepareInputTensor(cosFloat, cosInLocal_, cosCastBuf_, rope_size);
        AscendC::LocalTensor<float> sinFloat;
        PrepareInputTensor(sinFloat, sinInLocal_, sinCastBuf_, rope_size);

        // Allocate outputs
        vOutQueue_.AllocTensor<dataType>(vOutLocal_);
        kEmbedOutQueue_.AllocTensor<dataType>(kEmbedOutLocal_);

        AscendC::LocalTensor<float> vFloat;
        AscendC::LocalTensor<float> kEmbedFloat;
        PrepareOutputTensor(vFloat, vOutLocal_, vCastBuf_, rms_size);
        PrepareOutputTensor(kEmbedFloat, kEmbedOutLocal_, kEmbedCastBuf_, rope_size);

        // Compute RMSNorm
        ComputeRmsNorm(rmsInFloat, vFloat, rms_size);

        // Compute RoPE
        ComputeRoPE(ropeInFloat, cosFloat, sinFloat, kEmbedFloat, rope_size);

        // Finalize outputs (cast back from float)
        FinalizeOutputTensor(vOutLocal_, vFloat, rms_size);
        FinalizeOutputTensor(kEmbedOutLocal_, kEmbedFloat, rope_size);

        // Free input queues
        rmsInQueue_.FreeTensor(rmsInLocal_);
        ropeInQueue_.FreeTensor(ropeInLocal_);
        cosInQueue_.FreeTensor(cosInLocal_);
        sinInQueue_.FreeTensor(sinInLocal_);

        // EnQue outputs
        vOutQueue_.EnQue(vOutLocal_);
        kEmbedOutQueue_.EnQue(kEmbedOutLocal_);

        // Compute b, s, n from pos
        int32_t B = tiling_.B;
        int32_t N = tiling_.N;
        int32_t S = tiling_.S;
        int32_t b = pos / (S * N);
        int32_t rem = pos % (S * N);
        int32_t s = rem / N;
        int32_t n = rem % N;

        CopyOutOutputs(pos, b, s, n);
    }

    __aicore__ inline void ComputeRmsNorm(AscendC::LocalTensor<float> x,
                                          AscendC::LocalTensor<float> y, int32_t count)
    {
        AscendC::LocalTensor<float> xSq = reduceBuf_.Get<float>();
        AscendC::Mul(xSq, x, x, count);
        AscendC::PipeBarrier<PIPE_V>();

        float sumVal = 0.0f;
        for (int i = 0; i < count; ++i) {
            sumVal += xSq.GetValue(i);
        }

        float meanSq = sumVal * tiling_.invRmsSize + tiling_.eps;
        float invRmsVal = 1.0f / sqrt(meanSq);

        AscendC::LocalTensor<float> invRms = invRmsBuf_.Get<float>();
        AscendC::Duplicate(invRms, invRmsVal, 1);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Muls(y, x, invRmsVal, count);
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::Mul(y, y, gammaFloatLocal_, count);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeRoPE(AscendC::LocalTensor<float> ropeIn,
                                       AscendC::LocalTensor<float> cosLocal,
                                       AscendC::LocalTensor<float> sinLocal,
                                       AscendC::LocalTensor<float> kEmbedOut,
                                       int32_t rope_size)
    {
        int32_t half = rope_size / 2;
        AscendC::LocalTensor<float> rotateHalf = rotateHalfFloatBuf_.Get<float>();

        // rotate_half: [-k2, k1] using element-wise operations
        for (int i = 0; i < half; ++i) {
            rotateHalf.SetValue(i, -ropeIn.GetValue(half + i));
        }
        for (int i = 0; i < half; ++i) {
            rotateHalf.SetValue(half + i, ropeIn.GetValue(i));
        }
        AscendC::PipeBarrier<PIPE_V>();

        // k_embed = k * cos + rotate_half * sin
        // First half: k1 * cos1 + (-k2) * sin1
        for (int i = 0; i < half; ++i) {
            float k1 = ropeIn.GetValue(i);
            float c1 = cosLocal.GetValue(i);
            float s1 = sinLocal.GetValue(i);
            float rh = rotateHalf.GetValue(i);
            kEmbedOut.SetValue(i, k1 * c1 + rh * s1);
        }
        // Second half: k2 * cos2 + k1 * sin2
        for (int i = 0; i < half; ++i) {
            float k2 = ropeIn.GetValue(half + i);
            float c2 = cosLocal.GetValue(half + i);
            float s2 = sinLocal.GetValue(half + i);
            float rh = rotateHalf.GetValue(half + i);
            kEmbedOut.SetValue(half + i, k2 * c2 + rh * s2);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void UpdateCache(int32_t b, int32_t s, int32_t n,
                                       AscendC::LocalTensor<dataType> vLocal,
                                       AscendC::LocalTensor<dataType> kEmbedLocal)
    {
        int32_t mode = tiling_.cache_mode;

        if (mode == 0) {
            UpdateCacheNorm(b, s, n, vLocal, kEmbedLocal);
        } else if (mode == 1 || mode == 2) {
            UpdateCachePA(b, s, n, vLocal, kEmbedLocal);
        } else if (mode == 3) {
            UpdateCachePANZ(b, s, n, vLocal, kEmbedLocal);
        } else if (mode == 4) {
            UpdateCachePABlkBNSD(b, s, n, vLocal, kEmbedLocal);
        } else if (mode == 5) {
            UpdateCachePABlkNZ(b, s, n, vLocal, kEmbedLocal);
        }
    }

    __aicore__ inline void UpdateCacheNorm(int32_t b, int32_t s, int32_t n,
                                           AscendC::LocalTensor<dataType> vLocal,
                                           AscendC::LocalTensor<dataType> kEmbedLocal)
    {
        int32_t rms_size = tiling_.rms_size;
        int32_t rope_size = tiling_.rope_size;
        int32_t N = tiling_.N;
        int32_t S = tiling_.S;
        int32_t max_seq = tiling_.k_cache_dim2;
        int32_t B = tiling_.B;

        int64_t idx = 0;
        if (tiling_.index_numel == B * S) {
            idx = indexGM_.GetValue(b * S + s);
        } else {
            int pos = b * S + s;
            if (pos < tiling_.index_numel) {
                idx = indexGM_.GetValue(pos);
            }
        }
        if (idx < 0 || idx >= max_seq) {
            return;
        }

        int64_t kOffset = ((int64_t)b * N * max_seq + n * max_seq + idx) * rope_size;
        StoreUbToGm(kCacheOutGM_[kOffset], kEmbedLocal, rope_size);

        int64_t vOffset = ((int64_t)b * N * max_seq + n * max_seq + idx) * rms_size;
        StoreUbToGm(ckvCacheOutGM_[vOffset], vLocal, rms_size);
    }

    __aicore__ inline void UpdateCachePA(int32_t b, int32_t s, int32_t n,
                                         AscendC::LocalTensor<dataType> vLocal,
                                         AscendC::LocalTensor<dataType> kEmbedLocal)
    {
        int32_t rms_size = tiling_.rms_size;
        int32_t rope_size = tiling_.rope_size;
        int32_t N = tiling_.N;
        int32_t S = tiling_.S;
        int pos = b * S + s;
        if (pos >= tiling_.index_numel) {
            return;
        }
        int64_t idx = indexGM_.GetValue(pos);
        if (idx < 0) {
            return;
        }

        int64_t kCacheFlatSize = (int64_t)tiling_.k_cache_dim0 * tiling_.k_cache_dim1 *
                                 tiling_.k_cache_dim2 * tiling_.k_cache_dim3;
        int64_t cacheKN = kCacheFlatSize / rope_size;
        if (idx >= cacheKN) {
            return;
        }

        int64_t kOffset = idx * rope_size;
        StoreUbToGm(kCacheOutGM_[kOffset], kEmbedLocal, rope_size);

        int64_t ckvCacheFlatSize = (int64_t)tiling_.ckv_cache_dim0 * tiling_.ckv_cache_dim1 *
                                   tiling_.ckv_cache_dim2 * tiling_.ckv_cache_dim3;
        int64_t cacheVN = ckvCacheFlatSize / rms_size;
        if (idx >= cacheVN) {
            return;
        }
        int64_t vOffset = idx * rms_size;
        StoreUbToGm(ckvCacheOutGM_[vOffset], vLocal, rms_size);
    }

    __aicore__ inline void UpdateCachePANZ(int32_t b, int32_t s, int32_t n,
                                           AscendC::LocalTensor<dataType> vLocal,
                                           AscendC::LocalTensor<dataType> kEmbedLocal)
    {
        int32_t rms_size = tiling_.rms_size;
        int32_t rope_size = tiling_.rope_size;
        int32_t N = tiling_.N;
        int32_t S = tiling_.S;
        int pos = b * S + s;
        if (pos >= tiling_.index_numel) {
            return;
        }
        int64_t idx = indexGM_.GetValue(pos);
        if (idx < 0) {
            return;
        }

        int32_t block_size = tiling_.k_cache_dim1;
        int32_t dk = tiling_.k_cache_dim3;
        int32_t dv = tiling_.ckv_cache_dim3;
        int32_t dk0 = 16;
        int32_t dv0 = 16;
        int32_t dk1 = dk / dk0;
        int32_t dv1 = dv / dv0;
        int32_t bn = tiling_.k_cache_dim0;

        int64_t bn_id = idx / block_size;
        int64_t block_offset = idx % block_size;
        if (bn_id >= bn) {
            return;
        }

        AscendC::LocalTensor<dataType> scatterK = scatterKBuf_.Get<dataType>();
        for (int d = 0; d < dk1; ++d) {
            int64_t offset = ((bn_id * N * dk1 + n * dk1 + d) * block_size + block_offset) * dk0;
            for (int i = 0; i < dk0; ++i) {
                scatterK.SetValue(i, kEmbedLocal.GetValue(d * dk0 + i));
            }
            StoreUbToGm(kCacheOutGM_[offset], scatterK, dk0);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        AscendC::LocalTensor<dataType> scatterV = scatterVBuf_.Get<dataType>();
        for (int d = 0; d < dv1; ++d) {
            int64_t offset = ((bn_id * N * dv1 + n * dv1 + d) * block_size + block_offset) * dv0;
            for (int i = 0; i < dv0; ++i) {
                scatterV.SetValue(i, vLocal.GetValue(d * dv0 + i));
            }
            StoreUbToGm(ckvCacheOutGM_[offset], scatterV, dv0);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
    }

    __aicore__ inline void UpdateCachePABlkBNSD(int32_t b, int32_t s, int32_t n,
                                                AscendC::LocalTensor<dataType> vLocal,
                                                AscendC::LocalTensor<dataType> kEmbedLocal)
    {
        int32_t rms_size = tiling_.rms_size;
        int32_t rope_size = tiling_.rope_size;
        int32_t S = tiling_.S;
        int32_t block_size = tiling_.k_cache_dim1;
        int32_t ceil_div_s = (S + block_size - 1) / block_size;
        int32_t seq_id = s / block_size;
        int32_t seq_start = seq_id * block_size;
        int32_t idx_pos = b * ceil_div_s + seq_id;
        if (idx_pos >= tiling_.index_numel) {
            return;
        }
        int64_t idx_val = indexGM_.GetValue(idx_pos);
        if (idx_val < 0) {
            return;
        }
        int64_t cache_b = idx_val / block_size;
        if (cache_b >= tiling_.k_cache_dim0) {
            return;
        }

        int32_t offset_in_block = s - seq_start;
        int64_t kOffset = ((cache_b * block_size + offset_in_block) * tiling_.N + n) * rope_size;
        int64_t vOffset = ((cache_b * block_size + offset_in_block) * tiling_.N + n) * rms_size;
        StoreUbToGm(kCacheOutGM_[kOffset], kEmbedLocal, rope_size);
        StoreUbToGm(ckvCacheOutGM_[vOffset], vLocal, rms_size);
    }

    __aicore__ inline void UpdateCachePABlkNZ(int32_t b, int32_t s, int32_t n,
                                              AscendC::LocalTensor<dataType> vLocal,
                                              AscendC::LocalTensor<dataType> kEmbedLocal)
    {
        int32_t rms_size = tiling_.rms_size;
        int32_t rope_size = tiling_.rope_size;
        int32_t S = tiling_.S;
        int32_t block_size = tiling_.k_cache_dim1;
        int32_t ceil_div_s = (S + block_size - 1) / block_size;
        int32_t seq_id = s / block_size;
        int32_t seq_start = seq_id * block_size;
        int32_t idx_pos = b * ceil_div_s + seq_id;
        if (idx_pos >= tiling_.index_numel) {
            return;
        }
        int64_t idx_val = indexGM_.GetValue(idx_pos);
        if (idx_val < 0) {
            return;
        }
        int64_t cache_b = idx_val / block_size;
        int32_t bn = tiling_.k_cache_dim0;
        if (cache_b >= bn) {
            return;
        }

        int32_t dk = tiling_.k_cache_dim3;
        int32_t dv = tiling_.ckv_cache_dim3;
        int32_t dk0 = 16;
        int32_t dv0 = 16;
        int32_t dk1 = dk / dk0;
        int32_t dv1 = dv / dv0;
        int32_t offset_in_block = s - seq_start;

        AscendC::LocalTensor<dataType> scatterK = scatterKBuf_.Get<dataType>();
        for (int d = 0; d < dk1; ++d) {
            int64_t offset = ((cache_b * tiling_.N * dk1 + n * dk1 + d) * block_size + offset_in_block) * dk0;
            for (int i = 0; i < dk0; ++i) {
                scatterK.SetValue(i, kEmbedLocal.GetValue(d * dk0 + i));
            }
            StoreUbToGm(kCacheOutGM_[offset], scatterK, dk0);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
        AscendC::LocalTensor<dataType> scatterV = scatterVBuf_.Get<dataType>();
        for (int d = 0; d < dv1; ++d) {
            int64_t offset = ((cache_b * tiling_.N * dv1 + n * dv1 + d) * block_size + offset_in_block) * dv0;
            for (int i = 0; i < dv0; ++i) {
                scatterV.SetValue(i, vLocal.GetValue(d * dv0 + i));
            }
            StoreUbToGm(ckvCacheOutGM_[offset], scatterV, dv0);
            AscendC::PipeBarrier<PIPE_MTE2>();
        }
    }

    KvRmsnormRopeCacheTiling tiling_;
    AscendC::TPipe *pipe_;

    AscendC::GlobalTensor<dataType> kvGM_;
    AscendC::GlobalTensor<dataType> gammaGM_;
    AscendC::GlobalTensor<dataType> cosGM_;
    AscendC::GlobalTensor<dataType> sinGM_;
    AscendC::GlobalTensor<int64_t> indexGM_;
    AscendC::GlobalTensor<dataType> kCacheGM_;
    AscendC::GlobalTensor<dataType> ckvCacheGM_;
    AscendC::GlobalTensor<dataType> kCacheOutGM_;
    AscendC::GlobalTensor<dataType> ckvCacheOutGM_;
    AscendC::GlobalTensor<dataType> kEmbedOutGM_;
    AscendC::GlobalTensor<dataType> vOutGM_;

    // Input queues
    AscendC::TQue<AscendC::TPosition::VECIN, 0> gammaInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> rmsInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> ropeInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> cosInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> sinInQueue_;

    // Output queues
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> vOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> kEmbedOutQueue_;

    // Compute buffers (TBuf)
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> invRmsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ropeInFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rotateHalfFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kEmbedFloatBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scatterKBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scatterVBuf_;

    // Cast buffers (TBuf)
    AscendC::TBuf<AscendC::TPosition::VECCALC> rmsCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ropeCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> cosCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sinCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> vCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> kEmbedCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaCastBuf_;

    // LocalTensor members
    AscendC::LocalTensor<dataType> rmsInLocal_;
    AscendC::LocalTensor<dataType> ropeInLocal_;
    AscendC::LocalTensor<dataType> cosInLocal_;
    AscendC::LocalTensor<dataType> sinInLocal_;
    AscendC::LocalTensor<dataType> vOutLocal_;
    AscendC::LocalTensor<dataType> kEmbedOutLocal_;
    AscendC::LocalTensor<float> gammaFloatLocal_;
};
