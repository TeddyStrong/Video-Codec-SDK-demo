# Video-Codec-SDK-demo

This demo utilizes nvidia's Video Codec SDK and CUDA to realize GPU decoding and encoding acceleration.

## Dependencies
This demo is tested with WSL2 Ubuntu 18.04. Other required settings are as follows. Note the version used in this demo.
* gcc 6.5.0
* CUDA 11.1
* FFmpeg 4.4
* CMake 3.10.2

## Build
### Build with CMake
````
mkdir bin
mkdir build && cd build
cmake ..
make && cd ..
````

## Test
````
mkdir videos && mkdir outputs
````
Place your own video file in ./videos, and change the file paths in ```int main()``` in ./src/demo.cpp.
### Run
````
cd bin && ./gpu_decoder_demo $$ cd ../outputs
````
The results are in ./outputs.