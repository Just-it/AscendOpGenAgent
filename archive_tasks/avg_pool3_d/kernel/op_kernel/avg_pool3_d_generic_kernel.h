#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

#include "kernel_common.h"
#include "avg_pool3_d_tiling.h"

class AvgPool3DGenericKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(x), static_cast<uint64_t>(tiling_.N) * tiling_.inSpatial * tiling_.C);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(y), static_cast<uint64_t>(tiling_.N) * tiling_.outSpatial * tiling_.C);

        if ASCEND_IS_AIV {
            pipe_ = pipe;
            vecLen_ = tiling_.vectorLen > 0 ? tiling_.vectorLen : tiling_.C;
            if (vecLen_ > tiling_.C) {
                vecLen_ = tiling_.C;
            }

            pipe_->InitBuffer(inQueue_, 1, static_cast<uint32_t>(vecLen_ * sizeof(float)));
            pipe_->InitBuffer(outQueue_, 1, static_cast<uint32_t>(vecLen_ * sizeof(float)));
            pipe_->InitBuffer(accBuf_, static_cast<uint32_t>(vecLen_ * sizeof(float)));
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int workerId = static_cast<int>(AscendC::GetBlockIdx());
            int workerCount = tiling_.launchBlocks;
            if (workerCount <= 0) {
                workerCount = 1;
            }

            const int totalRows = tiling_.N * tiling_.outSpatial;
            for (int outRow = workerId; outRow < totalRows; outRow += workerCount) {
                ProcessOneRow(outRow);
            }
        }
    }

private:
    __aicore__ inline int ComputeDivisor(int od, int oh, int ow) const
    {
        if (tiling_.divisorOverride > 0) {
            return tiling_.divisorOverride;
        }
        if (tiling_.countIncludePad > 0) {
            return tiling_.kD * tiling_.kH * tiling_.kW;
        }

        const int dStart = od * tiling_.sD - tiling_.pD;
        const int hStart = oh * tiling_.sH - tiling_.pH;
        const int wStart = ow * tiling_.sW - tiling_.pW;

        const int dBegin = dStart < 0 ? 0 : dStart;
        const int hBegin = hStart < 0 ? 0 : hStart;
        const int wBegin = wStart < 0 ? 0 : wStart;

        int dEnd = dStart + tiling_.kD;
        int hEnd = hStart + tiling_.kH;
        int wEnd = wStart + tiling_.kW;

        if (dEnd > tiling_.D) {
            dEnd = tiling_.D;
        }
        if (hEnd > tiling_.H) {
            hEnd = tiling_.H;
        }
        if (wEnd > tiling_.W) {
            wEnd = tiling_.W;
        }

        const int validD = dEnd > dBegin ? (dEnd - dBegin) : 0;
        const int validH = hEnd > hBegin ? (hEnd - hBegin) : 0;
        const int validW = wEnd > wBegin ? (wEnd - wBegin) : 0;
        return validD * validH * validW;
    }

    __aicore__ inline void ProcessOneRowChunk(int nIdx, int od, int oh, int ow, int outRow, int cBase, int cLen)
    {
        accLocal_ = accBuf_.Get<float>();
        AscendC::Duplicate(accLocal_, 0.0f, cLen);

        const int kwGroups = static_cast<int>(CeilDivU32(static_cast<uint32_t>(tiling_.kW), static_cast<uint32_t>(tiling_.kW)));
        for (int kd = 0; kd < tiling_.kD; ++kd) {
            const int idVal = od * tiling_.sD - tiling_.pD + kd;
            if (idVal < 0 || idVal >= tiling_.D) {
                continue;
            }
            for (int kh = 0; kh < tiling_.kH; ++kh) {
                const int ihVal = oh * tiling_.sH - tiling_.pH + kh;
                if (ihVal < 0 || ihVal >= tiling_.H) {
                    continue;
                }
                for (int kwBase = 0; kwBase < kwGroups; ++kwBase) {
                    for (int kwLocal = 0; kwLocal < tiling_.kW; ++kwLocal) {
                        const int kw = kwBase * tiling_.kW + kwLocal;
                        if (kw >= tiling_.kW) {
                            continue;
                        }
                        const int iwVal = ow * tiling_.sW - tiling_.pW + kw;
                        if (iwVal < 0 || iwVal >= tiling_.W) {
                            continue;
                        }
                        const int inRow = nIdx * tiling_.inSpatial + idVal * tiling_.hw + ihVal * tiling_.W + iwVal;
                        inQueue_.AllocTensor<float>(inLocal_);
                        AscendC::DataCopy(inLocal_, xGM_[static_cast<uint64_t>(inRow) * tiling_.C + cBase], cLen);
                        inQueue_.EnQue(inLocal_);
                        inQueue_.DeQue<float>(inLocal_);
                        AscendC::Add(accLocal_, accLocal_, inLocal_, cLen);
                        inQueue_.FreeTensor(inLocal_);
                    }
                }
            }
        }

        outQueue_.AllocTensor<float>(outLocal_);
        const int divisor = ComputeDivisor(od, oh, ow);
        if (divisor <= 0) {
            AscendC::Duplicate(outLocal_, 0.0f, cLen);
        } else {
            const float invDiv = 1.0f / static_cast<float>(divisor);
            AscendC::Muls(outLocal_, accLocal_, invDiv, cLen);
        }

        outQueue_.EnQue(outLocal_);
        outQueue_.DeQue<float>(outLocal_);
        AscendC::DataCopy(yGM_[static_cast<uint64_t>(outRow) * tiling_.C + cBase], outLocal_, cLen);
        outQueue_.FreeTensor(outLocal_);
    }

    __aicore__ inline void ProcessOneRow(int outRow)
    {
        const int nIdx = outRow / tiling_.outSpatial;
        const int outRem = outRow - nIdx * tiling_.outSpatial;
        const int od = outRem / (tiling_.OH * tiling_.OW);
        const int odRem = outRem - od * (tiling_.OH * tiling_.OW);
        const int oh = odRem / tiling_.OW;
        const int ow = odRem - oh * tiling_.OW;

        for (int cBase = 0; cBase < tiling_.C; cBase += vecLen_) {
            const int remain = tiling_.C - cBase;
            const int cLen = remain < vecLen_ ? remain : vecLen_;
            ProcessOneRowChunk(nIdx, od, oh, ow, outRow, cBase, cLen);
        }
    }

private:
    AvgPool3DKernelTiling tiling_{};
    AscendC::TPipe *pipe_{nullptr};
    int vecLen_{0};

    AscendC::GlobalTensor<float> xGM_;
    AscendC::GlobalTensor<float> yGM_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf_;

    AscendC::LocalTensor<float> inLocal_;
    AscendC::LocalTensor<float> outLocal_;
    AscendC::LocalTensor<float> accLocal_;
};
