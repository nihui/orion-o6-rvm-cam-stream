# orion-o6-rvm-cam-stream

This is a sample project for real-time video segmentation of a USB camera and streaming it in jpg format over HTTP. It supports neural network inference on the CPU/GPU/NPU of Orion O6.

【orion o6 CPU GPU NPU 摄像头实时 RVM 人像分割和 http 推流】https://www.bilibili.com/video/BV1uN59zSEuf

The following steps should be performed on Orion O6.

## build and install ncnn

https://github.com/Tencent/ncnn/wiki/how-to-build#build-for-linux

```shell
git clone https://github.com/Tencent/ncnn.git
cd ncnn
mkdir build
cd build
cmake ..
make -j12
make install
```

## setup opencv-mobile

```shell
wget https://github.com/nihui/opencv-mobile/releases/latest/download/opencv-mobile-4.11.0-debian-bookworm-aarch64.zip
unzip -q opencv-mobile-4.11.0-debian-bookworm-aarch64.zip
```

## build this project

```shell
git clone https://github.com/nihui/orion-o6-rvm-cam-stream
cd orion-o6-rvm-cam-stream
mkdir build
cd build
cmake ..
make
```

## cam stream with browser

```shell
cd orion-o6-rvm-cam-stream/build
./test
```

This program will open usb camera (/dev/video3) ➡️ rvm (robust human video matting) portrait on npu ➡️ stream jpeg on http

Open the streaming url with browser, and you can see the rvm result

Sample output
```
[0 Mali-G720-Immortalis]  queueC=0[2]  queueG=0[2]  queueT=0[2]
[0 Mali-G720-Immortalis]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[0 Mali-G720-Immortalis]  fp16-p/s/u/a=1/1/1/1  int8-p/s/u/a=1/1/1/1
[0 Mali-G720-Immortalis]  subgroup=16(16~16)  ops=1/1/1/1/1/1/1/1/1/1
[0 Mali-G720-Immortalis]  fp16-8x8x16/16x8x8/16x8x16/16x16x16=0/0/0/0
[1 llvmpipe (LLVM 15.0.6, 128 bits)]  queueC=0[1]  queueG=0[1]  queueT=0[1]
[1 llvmpipe (LLVM 15.0.6, 128 bits)]  bugsbn1=0  bugbilz=0  bugcopc=0  bugihfa=0
[1 llvmpipe (LLVM 15.0.6, 128 bits)]  fp16-p/s/u/a=1/1/1/0  int8-p/s/u/a=1/1/1/1
[1 llvmpipe (LLVM 15.0.6, 128 bits)]  subgroup=4(4~4)  ops=1/1/1/1/1/1/0/1/0/0
[1 llvmpipe (LLVM 15.0.6, 128 bits)]  fp16-8x8x16/16x8x8/16x8x16/16x16x16=0/0/0/0
   devpath = /dev/video3
   driver = uvcvideo
   card = ESP UVC Device: 
   bus_info = usb-xhci-hcd.2.auto-1
   version = 6012c
   capabilities = 84a00001
   device_caps = 4200001
   fmt = Motion-JPEG  47504a4d
       size = 320 x 240   95.00
       size = 640 x 480   83.75
       size = 480 x 320   87.50
       size = 320 x 240   95.00
           fps = 1 / 15  ~   1 / 15  (+1 +15)   15.00
cap_pixelformat = 47504a4d  MJPG
cap_width = 320
cap_height = 240
bytesperline: 0
cap_numerator = 1
cap_denominator = 15
requestbuffers.count = 3
streaming at http://127.0.0.1:7766
streaming at http://192.168.1.7:7766
streaming at http://127.0.0.1:7766
opencv-mobile HW JPG encoder with v4l cix
client accepted 192.168.1.54 41256
```

## some note for main.cpp

change camera device index and resolution
```cpp
cv::VideoCapture cap;
cap.set(cv::CAP_PROP_FRAME_WIDTH, 240);
cap.set(cv::CAP_PROP_FRAME_HEIGHT, 240);
cap.open(3);
```

change http streaming port
```
cv::VideoWriter http;
http.open("httpjpg", 7766);
```

change rvm inference backend
```
// rvm_cpu.run(bgr_512, out);
// rvm_gpu.run(bgr_512, out);
rvm_npu.run(bgr_512, out);
```
