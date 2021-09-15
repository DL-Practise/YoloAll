# 更新：
有朋友反馈yolox的模型下载了之后，显示不出来，通过排查，是yolox更新了模型格式，导致新的模型格式不兼容。其实官方同时给出了新旧模型的下载地址。见下面的下载链接！按照下面的下载链接下载的模式可以正常使用。

# 背景：
在通用目标检测中，yolo是一个奇迹般的存在，生命力经久不衰，从yolov1，v2，v3，v4，v5到最近的yolox，都充分说明了yolo在学术界以及工业界的受欢迎程度。除了基线的频繁迭代更新之外，还有很多针对yolo的优化精简版本，如下表所示：
表1
在如此众多的yolo版本中，除了官方的评测数据之外，有没有更加直观的方式来对比各种不同版本的yolo性能呢？很早之前我就幻想着能有一个集成各种yolo的版本，能够对图片，视频，实时码流进行对比，于是就有了这个YoloAll的想法。希望能在一个software里面对同一张图片，同一段视频，或者同一个camera的码流进行对比分析。

# YoloAll演示：
下面是一段YoloAll的演示视频。目前支持yolo_v3, yolo_v5, yolox, yolo_fastest四个版本；并且支持从图片，视频(mp4)以及camera进行推理分析。
B站搜索“yoloall”


# YoloALL结构：
YoloAll使用pyqt设计，整体工程结构如下图所示，其中model_zoo下面放置了所有支持的yolo版本，包括里面的各种小版本。
目前支持4个版本的yolo，并且提供了简单的扩展方法，可以把你喜欢的yolo扩展进去。详情见“扩展模型”一节。
* Yolov3：https://github.com/eriklindernoren/PyTorch-YOLOv3
* Yolov5：https://github.com/ultralytics/yolov5
* Yolox: https://github.com/Megvii-BaseDetection/YOLOX
* Yolo-fastest-v2: https://github.com/dog-qiuqiu/Yolo-FastestV2
感谢以上开源项目以及作者，感谢你们带来了这么好的yolo项目。


# YoloAll使用：
## 下载源码
首先从地址：https://github.com/DL-Practise/YoloAll下载源码

## 安装依赖
所有的yolo版本均放在model_zoo目录下面，进入到model_zoo中的各个yolo版本中，执行pip install -r requirements.txt。安装各个yolo版本的依赖库。

## 下载预训练模型
当前的YoloAll对于每个yolo版本已经内置了一个最小模型，如下图所示。
如果您还想测试诸如yolov5_l， yolox_l等大型模型，可以从如下链接下载预训练模型，并放置在对应的目录下面：

### Yolov3
* https://pjreddie.com/media/files/yolov3.weights
下载完毕之后，将模型放在model_zoo/yolov3下面即可

### Yolov5
* https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt 
* https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt
* https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt
* https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt

下载完毕之后，将模型放在model_zoo/yolov5下面即可

### Yolox(使用yolox官方github的Legacy models下面的链接，因此官方修改了模型格式，需要使用旧的模型格式)
* https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EW62gmO2vnNNs5npxjzunVwB9p307qqygaCkXdTO88BLUg?e=NMTQYw
* https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ERMTP7VFqrVBrXKMU7Vl4TcBQs0SUeCT7kvc-JdIbej4tQ?e=1MDo9y
* https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EWA8w_IEOzBKvuueBqfaZh0BeoG5sVzR-XYbOJO4YlOkRw?e=wHWOBE
* https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EdgVPHBziOVBtGAXHfeHI5kBza0q9yyueMGdT0wXZfI1rQ?e=tABO5u
* https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EZ-MV1r_fMFPkPrNjvbJEMoBLOLAnXH-XKEB77w8LhXL6Q?e=mf6wOc

下载完毕之后，将模型放在model_zoo/yolox下面即可


此时，重新打开界面，能够显示出所有的yolo版本


## 扩展模型
YoloAll提供了非常方便的接口来集成您自己的yolo版本。将您自己的yolo版本放置在model_zoo中。并提供一个api.py的模块，里面实现如下方法：
#获取支持的子类型接口。例如yolov5中支持yolov5_s；yolov5_m; yolov5_l等
* def get_support_models()

#创建模型。根据模型的子类型名称，以及设备类型创建模型
* def create_model(model_name='yolov5_s', dev='cpu')

#进行一次推理，传入的img_array为opencv读取的图片
* def inference(img_array)

# QQ群：552703875
