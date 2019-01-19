---
layout: post
title:  "碧蓝航线的Live2D提取与播放"
date:   2019-01-19 20:40:47 +0800
categories: game
tags: hack
---

* content
{:toc}

## 碧（窑）蓝（子）航线的Live2D提取与播放

这篇博文纯属是闲得蛋疼的产物（~~博文中的泥石流~~），借助了prefare大佬的[文章]，跟着大佬的教程自己走了一遍，效果拔群。




### 0x00 需要的工具

> 提取Unity资源

- [Unity Asset Bundle Extractor] (UABE)
- prefare大佬的[AzurLaneLive2DExtract]（可以从大佬的[文章]中得到）

> Live2D播放

- Cubism官网自带的[Live2D Cubism Viewer]
- 或者是能够自己改代码的[CubismNativeFramework]

我选择了后者，毕竟能动些小手脚

### 0x01 提取Unity资源
首先得从比例比例官网下个碧蓝的apk，装到手机/模拟器上，然后随便你游客登陆也好，用自己b站账号登陆也好，进到游戏主界面点击右上角的`设置`->`资源`->`Live2D资源更新`，下载全部的Live2D资源，下载完就可以退出游戏了。

随便掏出个文件浏览器，在SD卡的目录下找到`Android/data/com.bilibili.azurlane/files/AssetBundles/live2d`，把里面的文件全部copy出来，保存到电脑上。如下图所示。  
![](https://zhouxuebin.club:4433/data/azurlane_live2d_img_1.png)

运行UABE的`AssetBundleExtractor`，在`File`->`Open`依次打开上面的文件（如第一个就是`aidang_2`），然后它会提问是否要解压，选是，然后随便输入个文件名保存就ok了，覆盖掉原文件也是没问题的。如果嫌弃累的话，后面有个能够一键操作 ~~（站上去自己动）~~ 的python脚本，直接运行就好了。

然后把解压的文件拖动到大佬写的exe上就好了。（好一会儿脑子秀逗了还以为要点开exe才拖，结果看了大佬的代码之后才明白只要拖到文件资源管理器上就ok了）

在上面一步执行完后，生成了一个`live2d`文件夹，里面每个文件夹对应了每个live2d模型，如点开`aidang_2`，里面会有`aidang_2.moc3`、`aidang_2.model3.json`和`aidang_2.physics3.json`，以及`motions`和`textures`两个文件夹。  
![](https://zhouxuebin.club:4433/data/azurlane_live2d_img_2.png)  
![](https://zhouxuebin.club:4433/data/azurlane_live2d_img_3.png)

下面一步需要的就是这些文件了。

### 0x02 查看Live2D

最简单直接的方法，下载Cubism官网的[Live2D Cubism Viewer]并打开，把`moc3`文件或者`model3.json`文件拖到窗口上就能直接看了。在左边双击`motions`下的文件就可以播放对应的动画。  
![](https://zhouxuebin.club:4433/data/azurlane_live2d_img_4.png)

这样就ok了。

动起来就是这样子的了：

<video controls="controls" width="100%" height="100%" autoplay="autoplay">
<source src="https://zhouxuebin.club:4433/data/azurlane_live2d_demo.mp4" type="video/mp4" />
emmmm，你的浏览器似乎不支持html5，这样子就难办了，视频就看不到了~
</video>

如果有个大胆的想法（比如把右边的动画窗口嵌入到自己的一个程序界面上的话），就需要自己动动手了。


### 0x03 魔改Native demo
目标：实现一个不依赖Unity并且能够自主控制的播放窗口。 ~~其实就是想把它当成自己写的背景小程序中的一个插件而已啦~~


第一步，下载CubismCore：在[https://live2d.github.io/#native](https://live2d.github.io/#native)上点击下载`Download Cubism 3 SDK for Native betag`

第二步，下载GLEW：在[http://glew.sourceforge.net](http://glew.sourceforge.net)下载`Binaries Windows 32-bit and 64-bit`

第三步，下载GLFW：由于一些功能需要最新版(`3.3.0`)，但是预编译的只有`3.2.1`版本，所以要自己动手build，丰衣足食。
1. GitHub 从复制到粘贴：运行`git clone https://github.com/glfw/glfw`
2. 教科书般的cmake编译，我习惯将build文件夹设成`.../glfw/build`，把`CMAKE_INSTALL_PREFIX`设成`.../glfw/build/install`，打开VS，生成INSTALL，完事。

第四步，下载[CubismNativeSamples](https://github.com/Live2D/CubismNativeSamples/)：
1. `git clone --recursive https://github.com/Live2D/CubismNativeSamples`
2. ~~对cubism native framework进行教科书般cmake~~
3. ~~↑出事了，因为cmake脚本下的`include_directories("${FRAMEWORK_GLFW_PATH}")`和`include_directories("${FRAMEWORK_GLEW_PATH}")`这两行找不到~~

第五步，Build自己的项目，让cmake玩蛋去吧
1. 看了看代码，感觉也就这样吧，打开vs，新建一个空的c++项目，取名就叫`CubismBuild`吧，自己编译去
2. 把Framework下的`src`文件夹复制到这个项目（有.vcxproj文件）的文件夹下，右键`CubismBuild`项目，`添加`->`添加现有项`，除了`Rendering`下的源文件之外，把`src`下面的所有源文件都加进来，`Rendering`只要加`OpenGL`下的源文件和`CubismRenderer`两个文件就好了。
3. 在`CubismRenderer_OpenGLES2.hpp`第一行加上`#define CSM_TARGET_WIN_GL`，手动指定相应的cmake宏
4. 解压CubismCore和GLEW
5. 把`...\CubismNativeSamples\Samples\OpenGL\Demo\proj.win.cmake\Demo`下的所有文件也都一同复制到项目文件夹下，在项目中添加这些文件
6. 把`...\CubismNativeSamples\Samples\OpenGL\thirdParty\stb\include\stb_image.h`这个也复制并添加到项目中
7. 改一下编译参数，右键项目，点击`属性`，转到`VC++目录`，把上面复制过来的src、解压的Core、GLEW和编译过的GLFW的路径都添加一下  
   `包含路径`  
   ![](https://zhouxuebin.club:4433/data/azurlane_live2d_img_5.png)  
   `引用路径`和`库路径`  
   ![](https://zhouxuebin.club:4433/data/azurlane_live2d_img_6.png)
8. 然后在左边的`链接器`->`输入`->`附加依赖项`，加上`Live2DCubismCore_MDd.lib`，`opengl32.lib`，`glu32.lib`，`glew32.lib`和`glfw3.lib`。
9. 直接编译就ok了，运行的时候要把glew的dll复制过去，否则会提示dll缺失。
10. 改`LAppDefine.cpp`下的`ResourcesPath`，改成`CubismNativeSamples\Samples\Res`的绝对路径就大功告成了（路径用`/`分割，不要用`\`，最后一个`/`要保留）

**It's time to 魔改。**

有了源代码，全程debug一遍基本上就知道哪些代码在干哪些活了。

那个power和齿轮按钮没用，去掉。  
改下`LAppView`就好了

窗体背景透明，隐藏标题。  
改下`LAppDelegate`就好了，这里就是为什么要选择`3.3.0`的GLFW了，因为`3.2.1`没有更改窗口背景的API。

要得到窗体的handle  
在`LAppDelegate.cpp`加上
```cpp
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
```
调用`glfwGetWin32Window(GLFWwindow *)`就会返回一个hwnd，调用WinAPI把父窗体设置成自己的窗体就可以为所欲为了。

设置窗体大小  
在`LAppDefine.cpp`改就好了

更改模型的控制  
在`LAppModel`上改就好了

在main上面加一个命令行解析，大概也就这样吧。（第二行输出的就是GLFW窗体的hwnd，可以用于后续的窗体嵌入）  
![](https://zhouxuebin.club:4433/data/azurlane_live2d_img_7.png)

### 0x04 流水作业，解放双手

很麻烦对吧，要下很多东西对吧，其实要用到的只有UABE，大佬写的[AzurLaneLive2DExtract]而已，用一个python脚本执行足够了。

[下载脚本]（整合了[UABE]、[AzurLaneLive2DExtract]和魔改的native viewer）

> 你要做的：

自己动手把Unity资源从手机/模拟器复制到电脑上  
打开`process.py`，把上面的路径改成上面的文件夹路径就ok了

**运行：** `python process.py`  
自动解压Unity资源、提取Live2D

> 查看Live2D：

1. 使用Cubism 3自带的Viewer
2. 或在命令行敲`.\player\CubismBuild.exe -d 模型所在的文件夹`就好了
更多的参数（其实也就改了几个，可以通过敲`.\player\CubismBuild.exe`查看）  
（可能需要VC++ 2017的运行环境）

### 0xff 引用
- Prefare大佬的[文章]和代码[AzurLaneLive2DExtract]
- [Unity Asset Bundle Extractor]
- Cubism 3自带的viewer：[Live2D Cubism Viewer]

魔改的代码连自己都看不下去，就不开源了（遮脸）

最后：适度游戏益脑，沉迷游戏伤身（物理）


[文章]: https://www.perfare.net/1270.html
[Unity Asset Bundle Extractor]: https://7daystodie.com/forums/showthread.php?22675-Unity-Assets-Bundle-Extractor
[AzurLaneLive2DExtract]: https://github.com/Perfare/AzurLaneLive2DExtract
[Live2D Cubism Viewer]: https://www.live2d.com/ja/download
[CubismNativeFramework]: https://github.com/Live2D/CubismNativeFramework
[下载脚本]: https://zhouxuebin.club:4433/data/azurlane_live2d.zip
[UABE]: https://7daystodie.com/forums/showthread.php?22675-Unity-Assets-Bundle-Extractor