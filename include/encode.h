#pragma once

#include <fstream>
#include <iostream>
#include <memory>
#include <cuda.h>
#include "utils/NvCodecUtils.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "NvEncoder/NvEncoderOutputInVidMemCuda.h"
#include "utils/Logger.h"
#include "utils/NvEncoderCLIOptions.h"

class NvCUStream
{
public:
    NvCUStream(CUcontext cuDevice, int cuStreamType, std::unique_ptr<NvEncoderOutputInVidMemCuda> &pEnc)
    {
        device = cuDevice;
        CUDA_DRVAPI_CALL(cuCtxPushCurrent(device));

        // Create CUDA streams
        if (cuStreamType == 1)
        {
            ck(cuStreamCreate(&inputStream, CU_STREAM_DEFAULT));
            outputStream = inputStream;
        }
        else if (cuStreamType == 2)
        {
            ck(cuStreamCreate(&inputStream, CU_STREAM_DEFAULT));
            ck(cuStreamCreate(&outputStream, CU_STREAM_DEFAULT));
        }

        CUDA_DRVAPI_CALL(cuCtxPopCurrent(nullptr));

        // SSet input and output CUDA streams in driver
        pEnc->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR)&inputStream, (NV_ENC_CUSTREAM_PTR)&outputStream);
    }

    ~NvCUStream()
    {
        ck(cuCtxPushCurrent(device));

        if (inputStream == outputStream)
        {
            if (inputStream != nullptr)
                ck(cuStreamDestroy(inputStream));
        }
        else
        {
            if (inputStream != nullptr)
                ck(cuStreamDestroy(inputStream));

            if (outputStream != nullptr)
                ck(cuStreamDestroy(outputStream));
        }
        ck(cuCtxPopCurrent(nullptr));
    }

    CUstream GetOutputCUStream() { return outputStream; };
    CUstream GetInputCUStream() { return inputStream; };

private:
    CUcontext device;
    CUstream inputStream = nullptr, outputStream = nullptr;
};

class CRC
{
public:
    CRC(CUcontext cuDevice, uint32_t bufferSize)
    {
        device = cuDevice;

        ck(cuCtxPushCurrent(device));

        // Allocate video memory buffer to store CRC of encoded frame
        ck(cuMemAlloc(&crcVidMem, bufferSize));

        ck(cuCtxPopCurrent(nullptr));
    }

    ~CRC()
    {
        ck(cuCtxPushCurrent(device));

        ck(cuMemFree(crcVidMem));

        ck(cuCtxPopCurrent(nullptr));
    }

    void GetCRC(NV_ENC_OUTPUT_PTR pVideoMemBfr, CUstream outputStream)
    {
        ComputeCRC((uint8_t *)pVideoMemBfr, (uint32_t *)crcVidMem, outputStream);
    }

    CUdeviceptr GetCRCVidMemPtr() { return crcVidMem; };

private:
    CUcontext device;
    CUdeviceptr crcVidMem = 0;
};

// This class dumps the output - CRC and encoded stream, to a file.
// Output is first copied to host buffer and then dumped to a file.
class DumpVidMemOutput
{
public:
    DumpVidMemOutput(CUcontext cuDevice, uint32_t size, char *outFilePath, bool bUseCUStream)
    {
        device = cuDevice;
        bfrSize = size;
        bCRC = bUseCUStream;

        ck(cuCtxPushCurrent(device));

        // Allocate host memory buffer to copy encoded output and CRC
        ck(cuMemAllocHost((void **)&pHostMemEncOp, (bfrSize + (bCRC ? 4 : 0))));

        ck(cuCtxPopCurrent(NULL));

        // Open file to dump CRC
        if (bCRC)
        {
            crcFile = std::string(outFilePath) + "_crc.txt";
            fpCRCOut.open(crcFile, std::ios::out);
            pHostMemCRC = (uint32_t *)((uint8_t *)pHostMemEncOp + bfrSize);
        }
    }

    ~DumpVidMemOutput()
    {
        ck(cuCtxPushCurrent(device));

        ck(cuMemFreeHost(pHostMemEncOp));

        ck(cuCtxPopCurrent(NULL));

        if (bCRC)
        {
            fpCRCOut.close();
            std::cout << "CRC saved in file: " << crcFile << std::endl;
        }
    }

    void DumpOutputToFile(CUdeviceptr pEncFrameBfr, CUdeviceptr pCRCBfr, std::ofstream &fpOut, uint32_t nFrame)
    {
        ck(cuCtxPushCurrent(device));

        // Copy encoded frame from video memory buffer to host memory buffer
        ck(cuMemcpyDtoH(pHostMemEncOp, pEncFrameBfr, bfrSize));

        // Copy encoded frame CRC from video memory buffer to host memory buffer
        if (bCRC)
        {
            ck(cuMemcpyDtoH(pHostMemCRC, pCRCBfr, 4));
        }

        ck(cuCtxPopCurrent(NULL));

        // Write encoded bitstream in file
        uint32_t offset = sizeof(NV_ENC_ENCODE_OUT_PARAMS);
        uint32_t bitstream_size = ((NV_ENC_ENCODE_OUT_PARAMS *)pHostMemEncOp)->bitstreamSizeInBytes;
        uint8_t *ptr = pHostMemEncOp + offset;

        fpOut.write((const char *)ptr, bitstream_size);

        // Write CRC in file
        if (bCRC)
        {
            if (!nFrame)
            {
                fpCRCOut << "Frame num" << std::setw(10) << "CRC" << std::endl;
            }
            fpCRCOut << std::dec << std::setfill(' ') << std::setw(5) << nFrame << "          ";
            fpCRCOut << std::hex << std::setfill('0') << std::setw(8) << *pHostMemCRC << std::endl;
        }
    }

private:
    CUcontext device;
    uint32_t bfrSize;
    uint8_t *pHostMemEncOp = NULL;
    uint32_t *pHostMemCRC = NULL;
    bool bCRC;
    std::string crcFile;
    std::ofstream fpCRCOut;
};

void ShowEncoderCapability()
{
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    std::cout << "Encoder Capability" << std::endl
              << std::endl;
    for (int iGpu = 0; iGpu < nGpu; iGpu++)
    {
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));
        NvEncoderCuda enc(cuContext, 1280, 720, NV_ENC_BUFFER_FORMAT_NV12);

        std::cout << "GPU " << iGpu << " - " << szDeviceName << std::endl
                  << std::endl;
        std::cout << "\tH264:\t\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no") << std::endl
                  << "\tH264_444:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no") << std::endl
                  << "\tH264_ME:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no") << std::endl
                  << "\tH264_WxH:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_WIDTH_MAX)) << "*" << (enc.GetCapabilityValue(NV_ENC_CODEC_H264_GUID, NV_ENC_CAPS_HEIGHT_MAX)) << std::endl
                  << "\tHEVC:\t\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORTED_RATECONTROL_MODES) ? "yes" : "no") << std::endl
                  << "\tHEVC_Main10:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_10BIT_ENCODE) ? "yes" : "no") << std::endl
                  << "\tHEVC_Lossless:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_LOSSLESS_ENCODE) ? "yes" : "no") << std::endl
                  << "\tHEVC_SAO:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_SAO) ? "yes" : "no") << std::endl
                  << "\tHEVC_444:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_YUV444_ENCODE) ? "yes" : "no") << std::endl
                  << "\tHEVC_ME:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_SUPPORT_MEONLY_MODE) ? "yes" : "no") << std::endl
                  << "\tHEVC_WxH:\t"
                  << "  " << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_WIDTH_MAX)) << "*" << (enc.GetCapabilityValue(NV_ENC_CODEC_HEVC_GUID, NV_ENC_CAPS_HEIGHT_MAX)) << std::endl;

        std::cout << std::endl;

        enc.DestroyEncoder();
        ck(cuCtxDestroy(cuContext));
    }
}

void ShowHelpAndExitEncode(const char *szBadOption = NULL)
{
    bool bThrowError = false;
    std::ostringstream oss;
    if (szBadOption)
    {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    oss << "Options:" << std::endl
        << "-i               Input file path" << std::endl
        << "-o               Output file path" << std::endl
        << "-s               Input resolution in this form: WxH" << std::endl
        << "-if              Input format: iyuv nv12 yuv444 p010 yuv444p16 bgra bgra10 ayuv abgr abgr10" << std::endl
        << "-gpu             Ordinal of GPU to use" << std::endl
        << "-outputInVidMem  Set this to 1 to enable output in Video Memory" << std::endl
        << "-cuStreamType    Use CU stream for pre and post processing when outputInVidMem is set to 1" << std::endl
        << "                 CRC of encoded frames will be computed and dumped to file with suffix '_crc.txt' added" << std::endl
        << "                 to file specified by -o option " << std::endl
        << "                 0 : both pre and post processing are on NULL CUDA stream" << std::endl
        << "                 1 : both pre and post processing are on SAME CUDA stream" << std::endl
        << "                 2 : both pre and post processing are on DIFFERENT CUDA stream" << std::endl;
    oss << NvEncoderInitParam().GetHelpMessage() << std::endl;
    if (bThrowError)
    {
        throw std::invalid_argument(oss.str());
    }
    else
    {
        std::cout << oss.str();
        ShowEncoderCapability();
        exit(0);
    }
}

void ParseCommandLineEncode(int argc, char *argv[], char *szInputFileName, int &nWidth, int &nHeight,
                            NV_ENC_BUFFER_FORMAT &eFormat, char *szOutputFileName, NvEncoderInitParam &initParam, int &iGpu,
                            bool &bOutputInVidMem, int32_t &cuStreamType)
{
    std::ostringstream oss;
    int i;
    for (i = 1; i < argc; i++)
    {
        if (!_stricmp(argv[i], "-h"))
        {
            ShowHelpAndExitEncode();
        }
        if (!_stricmp(argv[i], "-i"))
        {
            if (++i == argc)
            {
                ShowHelpAndExitEncode("-i");
            }
            sprintf(szInputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-o"))
        {
            if (++i == argc)
            {
                ShowHelpAndExitEncode("-o");
            }
            sprintf(szOutputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-s"))
        {
            if (++i == argc || 2 != sscanf(argv[i], "%dx%d", &nWidth, &nHeight))
            {
                ShowHelpAndExitEncode("-s");
            }
            continue;
        }
        std::vector<std::string> vszFileFormatName =
            {
                "iyuv", "nv12", "yv12", "yuv444", "p010", "yuv444p16", "bgra", "bgra10", "ayuv", "abgr", "abgr10"};
        NV_ENC_BUFFER_FORMAT aFormat[] =
            {
                NV_ENC_BUFFER_FORMAT_IYUV,
                NV_ENC_BUFFER_FORMAT_NV12,
                NV_ENC_BUFFER_FORMAT_YV12,
                NV_ENC_BUFFER_FORMAT_YUV444,
                NV_ENC_BUFFER_FORMAT_YUV420_10BIT,
                NV_ENC_BUFFER_FORMAT_YUV444_10BIT,
                NV_ENC_BUFFER_FORMAT_ARGB,
                NV_ENC_BUFFER_FORMAT_ARGB10,
                NV_ENC_BUFFER_FORMAT_AYUV,
                NV_ENC_BUFFER_FORMAT_ABGR,
                NV_ENC_BUFFER_FORMAT_ABGR10,
            };
        if (!_stricmp(argv[i], "-if"))
        {
            if (++i == argc)
            {
                ShowHelpAndExitEncode("-if");
            }
            auto it = std::find(vszFileFormatName.begin(), vszFileFormatName.end(), argv[i]);
            if (it == vszFileFormatName.end())
            {
                ShowHelpAndExitEncode("-if");
            }
            eFormat = aFormat[it - vszFileFormatName.begin()];
            continue;
        }
        if (!_stricmp(argv[i], "-gpu"))
        {
            if (++i == argc)
            {
                ShowHelpAndExitEncode("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-outputInVidMem"))
        {
            if (++i == argc)
            {
                ShowHelpAndExitEncode("-outputInVidMem");
            }
            bOutputInVidMem = (atoi(argv[i]) != 0) ? true : false;
            continue;
        }
        if (!_stricmp(argv[i], "-cuStreamType"))
        {
            if (++i == argc)
            {
                ShowHelpAndExitEncode("-cuStreamType");
            }
            cuStreamType = atoi(argv[i]);
            continue;
        }

        // Regard as encoder parameter
        if (argv[i][0] != '-')
        {
            ShowHelpAndExitEncode(argv[i]);
        }
        oss << argv[i] << " ";
        while (i + 1 < argc && argv[i + 1][0] != '-')
        {
            oss << argv[++i] << " ";
        }
    }
    initParam = NvEncoderInitParam(oss.str().c_str());
}

template <class EncoderClass>
void InitializeEncoder(EncoderClass &pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};

    initializeParams.encodeConfig = &encodeConfig;
    pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);

    pEnc->CreateEncoder(&initializeParams);
}

void EncodeCuda(int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, NvEncoderInitParam encodeCLIOptions, CUcontext cuContext, std::ifstream &fpIn, std::ofstream &fpOut)
{
    std::unique_ptr<NvEncoderCuda> pEnc(new NvEncoderCuda(cuContext, nWidth, nHeight, eFormat));

    InitializeEncoder(pEnc, encodeCLIOptions, eFormat);

    int nFrameSize = pEnc->GetFrameSize();

    std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[nFrameSize]);
    int nFrame = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (true)
    {
        // Load the next frame from disk
        std::streamsize nRead = fpIn.read(reinterpret_cast<char *>(pHostFrame.get()), nFrameSize).gcount();
        // For receiving encoded packets
        std::vector<std::vector<uint8_t>> vPacket;
        if (nRead == nFrameSize)
        {
            const NvEncInputFrame *encoderInputFrame = pEnc->GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(cuContext, pHostFrame.get(), 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                                             (int)encoderInputFrame->pitch,
                                             pEnc->GetEncodeWidth(),
                                             pEnc->GetEncodeHeight(),
                                             CU_MEMORYTYPE_HOST,
                                             encoderInputFrame->bufferFormat,
                                             encoderInputFrame->chromaOffsets,
                                             encoderInputFrame->numChromaPlanes);

            pEnc->EncodeFrame(vPacket);
        }
        else
        {
            pEnc->EndEncode(vPacket);
        }
        nFrame += (int)vPacket.size();
        for (std::vector<uint8_t> &packet : vPacket)
        {
            // For each encoded packet
            fpOut.write(reinterpret_cast<char *>(packet.data()), packet.size());
        }

        if (nRead != nFrameSize)
            break;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Total time spent: %f ms\n", time);
    printf("Encode time spent per frame: %f ms\n", time / nFrame);

    pEnc->DestroyEncoder();

    std::cout << "Total frames encoded: " << nFrame << std::endl;
}

void EncodeCudaOpInVidMem(int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, NvEncoderInitParam encodeCLIOptions, CUcontext cuContext, std::ifstream &fpIn, std::ofstream &fpOut, char *outFilePath, int32_t cuStreamType)
{
    std::unique_ptr<NvEncoderOutputInVidMemCuda> pEnc(new NvEncoderOutputInVidMemCuda(cuContext, nWidth, nHeight, eFormat));

    InitializeEncoder(pEnc, encodeCLIOptions, eFormat);

    int nFrameSize = pEnc->GetFrameSize();
    bool bUseCUStream = cuStreamType != -1 ? true : false;

    std::unique_ptr<CRC> pCRC;
    std::unique_ptr<NvCUStream> pCUStream;
    if (bUseCUStream)
    {
        // Allocate CUDA streams
        pCUStream.reset(new NvCUStream(reinterpret_cast<CUcontext>(pEnc->GetDevice()), cuStreamType, pEnc));

        // When CUDA streams are used, the encoded frame's CRC is computed using cuda kernel
        pCRC.reset(new CRC(reinterpret_cast<CUcontext>(pEnc->GetDevice()), pEnc->GetOutputBufferSize()));
    }

    // For dumping output - encoded frame and CRC, to a file
    std::unique_ptr<DumpVidMemOutput> pDumpVidMemOutput(new DumpVidMemOutput(reinterpret_cast<CUcontext>(pEnc->GetDevice()), pEnc->GetOutputBufferSize(), outFilePath, bUseCUStream));

    std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[nFrameSize]);
    int nFrame = 0;

    // Encoding loop
    while (true)
    {
        // Load the next frame from disk
        std::streamsize nRead = fpIn.read(reinterpret_cast<char *>(pHostFrame.get()), nFrameSize).gcount();
        // For receiving encoded packets
        std::vector<NV_ENC_OUTPUT_PTR> pVideoMemBfr;

        if (nRead == nFrameSize)
        {
            const NvEncInputFrame *encoderInputFrame = pEnc->GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(cuContext, pHostFrame.get(), 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                                             (int)encoderInputFrame->pitch,
                                             pEnc->GetEncodeWidth(),
                                             pEnc->GetEncodeHeight(),
                                             CU_MEMORYTYPE_HOST,
                                             encoderInputFrame->bufferFormat,
                                             encoderInputFrame->chromaOffsets,
                                             encoderInputFrame->numChromaPlanes,
                                             false,
                                             bUseCUStream ? pCUStream->GetInputCUStream() : NULL);

            pEnc->EncodeFrame(pVideoMemBfr);
        }
        else
        {
            pEnc->EndEncode(pVideoMemBfr);
        }

        for (uint32_t i = 0; i < pVideoMemBfr.size(); ++i)
        {
            if (bUseCUStream)
            {
                // Compute CRC of encoded stream
                pCRC->GetCRC(pVideoMemBfr[i], pCUStream->GetOutputCUStream());
            }

            pDumpVidMemOutput->DumpOutputToFile((CUdeviceptr)(pVideoMemBfr[i]), bUseCUStream ? pCRC->GetCRCVidMemPtr() : 0, fpOut, nFrame);

            nFrame++;
        }

        if (nRead != nFrameSize)
            break;
    }

    pEnc->DestroyEncoder();

    std::cout << "Total frames encoded: " << nFrame << std::endl;
}

void EncodeCudaInVidMem(int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, NvEncoderInitParam encodeCLIOptions, CUcontext cuContext, std::ifstream &fpIn, std::ofstream &fpOut, char *outFilePath, int32_t cuStreamType)
{
    std::unique_ptr<NvEncoderOutputInVidMemCuda> pEnc(new NvEncoderOutputInVidMemCuda(cuContext, nWidth, nHeight, eFormat));

    InitializeEncoder(pEnc, encodeCLIOptions, eFormat);

    int nFrameSize = pEnc->GetFrameSize();
    bool bUseCUStream = cuStreamType != -1 ? true : false;

    std::unique_ptr<CRC> pCRC;
    std::unique_ptr<NvCUStream> pCUStream;
    if (bUseCUStream)
    {
        // Allocate CUDA streams
        pCUStream.reset(new NvCUStream(reinterpret_cast<CUcontext>(pEnc->GetDevice()), cuStreamType, pEnc));

        // When CUDA streams are used, the encoded frame's CRC is computed using cuda kernel
        pCRC.reset(new CRC(reinterpret_cast<CUcontext>(pEnc->GetDevice()), pEnc->GetOutputBufferSize()));
    }

    // For dumping output - encoded frame and CRC, to a file
    std::unique_ptr<DumpVidMemOutput> pDumpVidMemOutput(new DumpVidMemOutput(reinterpret_cast<CUcontext>(pEnc->GetDevice()), pEnc->GetOutputBufferSize(), outFilePath, bUseCUStream));

    std::unique_ptr<uint8_t[]> pHostFrame(new uint8_t[nFrameSize]);
    int nFrame = 0;

    // Encoding loop
    while (true)
    {
        // Load the next frame from disk
        std::streamsize nRead = fpIn.read(reinterpret_cast<char *>(pHostFrame.get()), nFrameSize).gcount();
        // For receiving encoded packets
        std::vector<NV_ENC_OUTPUT_PTR> pVideoMemBfr;

        if (nRead == nFrameSize)
        {
            const NvEncInputFrame *encoderInputFrame = pEnc->GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(cuContext, pHostFrame.get(), 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                                             (int)encoderInputFrame->pitch,
                                             pEnc->GetEncodeWidth(),
                                             pEnc->GetEncodeHeight(),
                                             CU_MEMORYTYPE_HOST,
                                             encoderInputFrame->bufferFormat,
                                             encoderInputFrame->chromaOffsets,
                                             encoderInputFrame->numChromaPlanes,
                                             false,
                                             bUseCUStream ? pCUStream->GetInputCUStream() : NULL);

            pEnc->EncodeFrame(pVideoMemBfr);
        }
        else
        {
            pEnc->EndEncode(pVideoMemBfr);
        }

        for (uint32_t i = 0; i < pVideoMemBfr.size(); ++i)
        {
            if (bUseCUStream)
            {
                // Compute CRC of encoded stream
                pCRC->GetCRC(pVideoMemBfr[i], pCUStream->GetOutputCUStream());
            }

            pDumpVidMemOutput->DumpOutputToFile((CUdeviceptr)(pVideoMemBfr[i]), bUseCUStream ? pCRC->GetCRCVidMemPtr() : 0, fpOut, nFrame);

            nFrame++;
        }

        if (nRead != nFrameSize)
            break;
    }

    pEnc->DestroyEncoder();

    std::cout << "Total frames encoded: " << nFrame << std::endl;
}