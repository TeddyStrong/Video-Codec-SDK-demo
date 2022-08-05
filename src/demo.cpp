#include "decode.h"
#include "encode.h"
#include "dec_enc.h"

#define DECODE_ONLY 0
#define DECODE_ENCODE 1
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

int main(int argc, char **argv)
{
    char szInFilePath[256] = "../videos/input.avi";
    char szOutRawFilePath[256] = "../outputs/output.yuv";
    char szOutFilePath[256] = "../outputs/output.avi";

    int iGpu = 0;
    Rect cropRect = {};
    Dim resizeDim = {};
    unsigned int opPoint = 0;
    bool bDispAllLayers = false;
    bool bOutPlanar = false;
    try
    {
        ck(cuInit(0));
        int nGpu = 0;
        ck(cuDeviceGetCount(&nGpu));
        if (iGpu < 0 || iGpu >= nGpu) {
            std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
            return 1;
        }
        CUdevice cuDevice = 0;
        ck(cuDeviceGet(&cuDevice, iGpu));
        char szDeviceName[80];
        ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        std::cout << "GPU in use: " << szDeviceName << std::endl;
        CUcontext cuContext = NULL;
        ck(cuCtxCreate(&cuContext, 0, cuDevice));

        CheckInputFile(szInFilePath);

#if DECODE_ONLY
        ParseCommandLine(argc, argv, szInFilePath, szOutRawFilePath, bOutPlanar, iGpu, cropRect, resizeDim, opPoint, bDispAllLayers);
        if (!*szOutRawFilePath)
        {
            // sprintf(szOutFilePath, bOutPlanar ? "../outputs/output.planar" : "../outputs/output.native");
            sprintf(szOutRawFilePath, "../outputs/output.yuv");
        }

        DecodeMediaFile(cuContext, szInFilePath, szOutRawFilePath, bOutPlanar, cropRect, resizeDim, opPoint, bDispAllLayers);
#elif DECODE_ENCODE
        int nOutBitDepth = 0;
        if (!*szOutFilePath)
        {
            sprintf(szOutFilePath, "../outputs/output.mp4");
        }
        DecodeEncodeMediafile(cuContext, szInFilePath, szOutFilePath, nOutBitDepth);
#endif
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
        exit(1);
    }

    return 0;
}