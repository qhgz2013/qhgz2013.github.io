---
layout: post
title:  "在VS2017上使用CUDA9.0编译OpenCV4.0.1"
date:   2019-01-18 14:09:31 +0800
categories: build
tags: computer-vision cmake
---

* content
{:toc}

为了卸载VS2015，把系统重装了一遍，再装上VS2017，结果再次编译OpenCV的时候居然出了点小毛病（主要还是CUDA9.0不兼容VS2017），在这里记录一下。



## 准备工作

- 在[OpenCV Release]上下载`4.0.1`的Source code
- 可选：在[OpenCV contrib Release]上下载`4.0.1`的Source code
- 在[cmake官网]上下载最新的cmake
- 在`Visual Studio Installer`上，确保`工作负载`下勾选了`使用C++的桌面开发`，并且在右边的`安装详细信息`一栏上，确保勾选了`适用于桌面的VC++ 2015.3 v14.00 (v140)工具集`

## 编译步骤

1. 解压OpenCV的源代码，假设解压到`D:\opencv-4.0.1`下
2. 可选：解压OpenCV contrib的源代码，假设解压到`D:\opencv_contrib-4.0.1`下
3. 安装cmake，运行cmake GUI
4. 在CMake GUI上，进行教科书般的编译设置：在`Where is the source code:`右边的文本框填入`D:/opencv-4.0.1`，在`Where to build the binaries:`填入`D:/opencv-4.0.1/build`。然后点击下面的`Configure`按钮，选择`Visual Studio 15 2017 Win64`，在下面的`Optional toolset to use`填入`v140,host=x64`。
5. 把`WITH_CUDA`勾上
6. 如果电脑上有python，同时也要给python装cv2包的话，把`BUILD_opencv_python3`勾上，否则去掉
7. 如果下载了opencv contrib，在`OPENCV_EXTRA_MODULES_PATH`填上`D:/opencv_contrib-4.0.1/modules`。
8. 再点次`Configure`，然后点`Generate`，最后点`Open Project`就可以在VS上打开该项目了
9. 要修改一处的源代码，否则编译CUDA时会出错：把`D:\opencv-4.0.1\modules\core\include\opencv2\core\cuda\detail\color_detail.hpp`下，第96-127行的`const`改成`constexpr`，参考了[这里]。
10. 在VS上把`Debug`改成`Release`，然后右击解决方案资源管理器的`INSTALL`，点击`生成`，然后干等就好了。
11. 生成的dll、lib和exe都在`D:\opencv-4.0.1\build\install`下，把它挪到一个好看点的位置，把bin文件夹添加到PATH就ok了。

[OpenCV Release]: https://github.com/opencv/opencv/releases
[OpenCV contrib Release]: https://github.com/opencv/opencv_contrib/releases
[cmake官网]: https://cmake.org/
[这里]: http://answers.opencv.org/question/205673/building-opencv-with-cuda-win10-vs-2017/
