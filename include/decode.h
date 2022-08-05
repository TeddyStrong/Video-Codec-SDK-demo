#pragma once

#include <iostream>
#include <algorithm>
#include <thread>
#include <cuda.h>
#include "NvDecoder/NvDecoder.h"
#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"
#include "AppDecUtils.h"
#include "ColorSpace.h"

void ConvertSemiplanarToPlanar(uint8_t *pHostFrame, int nWidth, int nHeight, int nBitDepth)
{
    if (nBitDepth == 8)
    {
        // nv12->iyuv
        YuvConverter<uint8_t> converter8(nWidth, nHeight);
        converter8.UVInterleavedToPlanar(pHostFrame);
    }
    else
    {
        // p016->yuv420p16
        YuvConverter<uint16_t> converter16(nWidth, nHeight);
        converter16.UVInterleavedToPlanar((uint16_t *)pHostFrame);
    }
}

/**
 *   @brief  Function to decode media file and write raw frames into an output file.
 *   @param  cuContext     - Handle to CUDA context
 *   @param  szInFilePath  - Path to file to be decoded
 *   @param  szOutFilePath - Path to output file into which raw frames are stored
 *   @param  bOutPlanar    - Flag to indicate whether output needs to be converted to planar format
 *   @param  cropRect      - Cropping rectangle coordinates
 *   @param  resizeDim     - Resizing dimensions for output
 *   @param  opPoint       - Select an operating point of an AV1 scalable bitstream
 *   @param  bDispAllLayers - Output all decoded frames of an AV1 scalable bitstream
 */
void DecodeMediaFile(CUcontext cuContext, const char *szInFilePath, const char *szOutFilePath, bool bOutPlanar,
                     const Rect &cropRect, const Dim &resizeDim, const unsigned int opPoint, const bool bDispAllLayers)
{
    std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
    if (!fpOut)
    {
        std::ostringstream err;
        err << "Unable to open output file: " << szOutFilePath << std::endl;
        throw std::invalid_argument(err.str());
    }

    FFmpegDemuxer demuxer(szInFilePath);
    NvDecoder dec(cuContext, false /*ture for gpu memory*/, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), false, false, &cropRect, &resizeDim);

    /* Set operating point for AV1 SVC. It has no impact for other profiles or codecs
     * PFNVIDOPPOINTCALLBACK Callback from video parser will pick operating point set to NvDecoder  */
    dec.SetOperatingPoint(opPoint, bDispAllLayers);

    int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
    uint8_t *pVideo = NULL, *pFrame;
    bool bDecodeOutSemiPlanar = false;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    do
    {
        demuxer.Demux(&pVideo, &nVideoBytes);
        nFrameReturned = dec.Decode(pVideo, nVideoBytes);
        if (!nFrame && nFrameReturned)
            LOG(INFO) << dec.GetVideoInfo();

        bDecodeOutSemiPlanar = (dec.GetOutputFormat() == cudaVideoSurfaceFormat_NV12) || (dec.GetOutputFormat() == cudaVideoSurfaceFormat_P016);

        for (int i = 0; i < nFrameReturned; i++)
        {
            pFrame = dec.GetFrame();
            if (bOutPlanar && bDecodeOutSemiPlanar)
            {
                ConvertSemiplanarToPlanar(pFrame, dec.GetWidth(), dec.GetHeight(), dec.GetBitDepth());
            }
            // dump YUV to disk
            if (dec.GetWidth() == dec.GetDecodeWidth())
            {
                fpOut.write(reinterpret_cast<char *>(pFrame), dec.GetFrameSize());
            }
            else
            {
                // 4:2:0 output width is 2 byte aligned. If decoded width is odd , luma has 1 pixel padding
                // Remove padding from luma while dumping it to disk
                // dump luma
                for (auto i = 0; i < dec.GetHeight(); i++)
                {
                    fpOut.write(reinterpret_cast<char *>(pFrame), dec.GetDecodeWidth() * dec.GetBPP());
                    pFrame += dec.GetWidth() * dec.GetBPP();
                }
                // dump Chroma
                fpOut.write(reinterpret_cast<char *>(pFrame), dec.GetChromaPlaneSize());
            }
        }
        nFrame += nFrameReturned;
    } while (nVideoBytes);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("Total time spent: %f ms\n", time);
    printf("Decode time spent per frame: %f ms\n", time / nFrame);

    std::vector<std::string> aszDecodeOutFormat = {"NV12", "P016", "YUV444", "YUV444P16"};
    if (bOutPlanar)
    {
        aszDecodeOutFormat[0] = "iyuv";
        aszDecodeOutFormat[1] = "yuv420p16";
    }
    std::cout << "Total frame decoded: " << nFrame << std::endl
              << "Saved in file " << szOutFilePath << " in "
              << aszDecodeOutFormat[dec.GetOutputFormat()]
              << " format" << std::endl;
    fpOut.close();
}

void ShowHelpAndExit(const char *szBadOption = NULL)
{
    bool bThrowError = false;
    std::ostringstream oss;
    if (szBadOption)
    {
        bThrowError = true;
        oss << "Error parsing \"" << szBadOption << "\"" << std::endl;
    }
    oss << "Options:" << std::endl
        << "-i             Input file path" << std::endl
        << "-o             Output file path" << std::endl
        << "-outplanar     Convert output to planar format" << std::endl
        << "-gpu           Ordinal of GPU to use" << std::endl
        << "-crop l,t,r,b  Crop rectangle in left,top,right,bottom (ignored for case 0)" << std::endl
        << "-resize WxH    Resize to dimension W times H (ignored for case 0)" << std::endl
        << "-oppoint n     Select an operating point of an AV1 scalable bitstream" << std::endl
        << "-alllayers     Output all decoded frames of an AV1 scalable bitstream" << std::endl;
    oss << std::endl;
    if (bThrowError)
    {
        throw std::invalid_argument(oss.str());
    }
    else
    {
        std::cout << oss.str();
        ShowDecoderCapability();
        exit(0);
    }
}

void ParseCommandLine(int argc, char *argv[], char *szInputFileName, char *szOutputFileName,
                      bool &bOutPlanar, int &iGpu, Rect &cropRect, Dim &resizeDim, unsigned int &opPoint, bool &bDispAllLayers)
{
    std::ostringstream oss;
    int i;
    bDispAllLayers = false;
    opPoint = 0;
    for (i = 1; i < argc; i++)
    {
        if (!_stricmp(argv[i], "-h"))
        {
            ShowHelpAndExit();
        }
        if (!_stricmp(argv[i], "-i"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-i");
            }
            sprintf(szInputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-o"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-o");
            }
            sprintf(szOutputFileName, "%s", argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-outplanar"))
        {
            bOutPlanar = true;
            continue;
        }
        if (!_stricmp(argv[i], "-gpu"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-gpu");
            }
            iGpu = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-crop"))
        {
            if (++i == argc || 4 != sscanf(
                                        argv[i], "%d,%d,%d,%d",
                                        &cropRect.l, &cropRect.t, &cropRect.r, &cropRect.b))
            {
                ShowHelpAndExit("-crop");
            }
            if ((cropRect.r - cropRect.l) % 2 == 1 || (cropRect.b - cropRect.t) % 2 == 1)
            {
                std::cout << "Cropping rect must have width and height of even numbers" << std::endl;
                exit(1);
            }
            continue;
        }
        if (!_stricmp(argv[i], "-resize"))
        {
            if (++i == argc || 2 != sscanf(argv[i], "%dx%d", &resizeDim.w, &resizeDim.h))
            {
                ShowHelpAndExit("-resize");
            }
            if (resizeDim.w % 2 == 1 || resizeDim.h % 2 == 1)
            {
                std::cout << "Resizing rect must have width and height of even numbers" << std::endl;
                exit(1);
            }
            continue;
        }
        if (!_stricmp(argv[i], "-oppoint"))
        {
            if (++i == argc)
            {
                ShowHelpAndExit("-oppoint");
            }
            opPoint = atoi(argv[i]);
            continue;
        }
        if (!_stricmp(argv[i], "-alllayers"))
        {

            bDispAllLayers = true;
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}