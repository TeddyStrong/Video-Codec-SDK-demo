#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <functional>
#include "NvEncoder/NvEncoderCuda.h"
#include "NvDecoder/NvDecoder.h"
#include "utils/NvCodecUtils.h"
#include "utils/NvEncoderCLIOptions.h"
#include "utils/FFmpegDemuxer.h"
#include "utils/ColorSpace.h"
#include "npp.h"

void DecodeEncodeMediafile(CUcontext cuContext, const char *szInFilePath, char *szOutFilePath, int nOutBitDepth = 0)
{
    using NvEncCudaPtr = std::unique_ptr<NvEncoderCuda, std::function<void(NvEncoderCuda *)>>;
    auto EncodeDeleteFunc = [](NvEncoderCuda *pEnc)
    {
        if (pEnc)
        {
            pEnc->DestroyEncoder();
            delete pEnc;
        }
    };

    // Delay instantiating the encoder until we've decoded some frames.
    NvEncCudaPtr pEnc(nullptr, EncodeDeleteFunc);
    NvEncoderInitParam encodeCLIOptions;

    std::ifstream fpIn(szInFilePath, std::ifstream::in | std::ifstream::binary);
    if (!fpIn)
    {
        std::ostringstream err;
        err << "Unable to open input file: " << szInFilePath << std::endl;
        throw std::invalid_argument(err.str());
    }

    std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
    if (!fpOut)
    {
        std::ostringstream err;
        err << "Unable to open output file: " << szOutFilePath << std::endl;
        throw std::invalid_argument(err.str());
    }

    // Prepare the demuxer and decoder for operations on the input stream.
    FFmpegDemuxer demuxer(szInFilePath);
    NvDecoder dec(cuContext, true, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), false, true);

    int nFrame = 0, nVideoBytes = 0, nFrameReturned = 0, iMatrix = 0;
    int nWidth = 0, nHeight = 0, nFrameSize = 0;
    uint8_t *pVideo = nullptr, *pFrame = 0;
    CUdeviceptr dpFrame = 0;
    bool bOut10 = false;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    CUdeviceptr pTmpImage_1 = 0;

    bool is_begin = true;
    do
    {
        demuxer.Demux(&pVideo, &nVideoBytes);
        nFrameReturned = dec.Decode(pVideo, nVideoBytes);
        if (!nFrame && nFrameReturned && is_begin == true)
        {
            LOG(INFO) << dec.GetVideoInfo();
            // Get output frame size from decoder
            nWidth = dec.GetWidth();
            nHeight = dec.GetHeight();
            nFrameSize = nWidth * nHeight * 3;
            cuMemAlloc(&pTmpImage_1, nWidth * nHeight * 3);
            is_begin = false;
        }

        for (int i = 0; i < nFrameReturned; i++)
        {
            iMatrix = dec.GetVideoFormatInfo().video_signal_description.matrix_coefficients;
            pFrame = dec.GetFrame();

            // Color switching. This is only for testing.
            Nv12ToNv12(pFrame, dec.GetWidth(), dec.GetWidth(), dec.GetHeight());

            if (!pEnc)
            {
                // Create an encoder after decoded some frames.

                bOut10 = nOutBitDepth ? nOutBitDepth > 8 : dec.GetBitDepth() > 8;
                pEnc.reset(new NvEncoderCuda(cuContext, dec.GetWidth(), dec.GetHeight(),
                                             bOut10 ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT : NV_ENC_BUFFER_FORMAT_NV12));

                NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
                NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
                initializeParams.encodeConfig = &encodeConfig;
                pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());

                // Video frame rate
                initializeParams.frameRateNum = 20;
                encodeCLIOptions.SetInitParams(&initializeParams, bOut10 ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT : NV_ENC_BUFFER_FORMAT_NV12);

                pEnc->CreateEncoder(&initializeParams);
            }

            std::vector<std::vector<uint8_t>> vPacket;
            const NvEncInputFrame *encoderInputFrame = pEnc->GetNextInputFrame();

            if ((bOut10 && dec.GetBitDepth() > 8) || (!bOut10 && dec.GetBitDepth() == 8))
            {
                NvEncoderCuda::CopyToDeviceFrame(cuContext,
                                                 pFrame,
                                                 dec.GetDeviceFramePitch(),
                                                 (CUdeviceptr)encoderInputFrame->inputPtr,
                                                 encoderInputFrame->pitch,
                                                 pEnc->GetEncodeWidth(),
                                                 pEnc->GetEncodeHeight(),
                                                 CU_MEMORYTYPE_DEVICE,
                                                 encoderInputFrame->bufferFormat,
                                                 encoderInputFrame->chromaOffsets,
                                                 encoderInputFrame->numChromaPlanes);
                pEnc->EncodeFrame(vPacket);
            }
            else
            {
                // Bit depth conversion is needed
                if (bOut10)
                {
                    ConvertUInt8ToUInt16((uint8_t *)pFrame, (uint16_t *)encoderInputFrame->inputPtr, dec.GetDeviceFramePitch(), encoderInputFrame->pitch,
                                         pEnc->GetEncodeWidth(),
                                         pEnc->GetEncodeHeight() + ((pEnc->GetEncodeHeight() + 1) / 2));
                }
                else
                {
                    ConvertUInt16ToUInt8((uint16_t *)pFrame, (uint8_t *)encoderInputFrame->inputPtr, dec.GetDeviceFramePitch(), encoderInputFrame->pitch,
                                         pEnc->GetEncodeWidth(),
                                         pEnc->GetEncodeHeight() + ((pEnc->GetEncodeHeight() + 1) / 2));
                }
                pEnc->EncodeFrame(vPacket);
            }
            nFrame += (int)vPacket.size();
            for (std::vector<uint8_t> &packet : vPacket)
            {
                // std::cout << packet.size() << "\t\r";
                fpOut.write(reinterpret_cast<char *>(packet.data()), packet.size());
            }
        }
    } while (nVideoBytes);

    if (pEnc)
    {
        std::vector<std::vector<uint8_t>> vPacket;
        pEnc->EndEncode(vPacket);
        nFrame += (int)vPacket.size();
        for (std::vector<uint8_t> &packet : vPacket)
        {
            // std::cout << packet.size() << "\t\r";
            fpOut.write(reinterpret_cast<char *>(packet.data()), packet.size());
        }
        std::cout << std::endl;
    }

    fpIn.close();
    fpOut.close();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Total frame decoded and encoded: " << nFrame << std::endl
              << "Saved in file " << szOutFilePath << " of " << (bOut10 ? 10 : 8) << " bit depth" << std::endl;
    printf("Total time spent: %f ms\n", time);
    printf("Decode and encode time spent per frame: %f ms\n", time / nFrame);
}