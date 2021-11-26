# 官方讨论群
QQ群：552703875  
微信群：15158106211(先加作者微信，再邀请入群)

# YoloAll项目简介
YoloAll是一个将当前主流Yolo版本集成到同一个UI界面下的推理预测工具。可以迅速切换不同的yolo版本，并且可以针对图片，视频，摄像头码流进行实时推理，可以很方便，直观的对比不同版本的yolo的推理效果，耗时等。  
自从YoloAll V1版本推出以后，得到了很多朋友的喜欢，大家也跟我一样，被众多的Yolo版本所困扰，有时为了比较两个不同版本的Yolo的效果，往往需要花费很长时间搭建环境，并且很容易出错，因此才有了这个开发这个YoloAll的想法，能够非常方便的测试不同Yolo的效果！注意这个是测试推理效果的哦，不能进行训练，如果想要非常方便的训练，推荐另外一个可视化的YoloX训练项目：YoloX EasyTrain：https://github.com/DL-Practise/YoloX_EasyTrain![image](https://user-images.githubusercontent.com/16218143/143548303-d7927b0d-edf5-4423-850a-6d466753d6eb.png)


# YoloAll V2.0新特性
	更全、更美、更强、更易用
* 增加了YoloV4，更新YoloV5到6.0版本，更新YoloX到0.1.1版本  
* 重新设计了界面，美观度提升，使用起来更加舒畅  
* 增加了模型推理时候的参数配置  
* 增加了模型自动下载功能，方便下载与训练模型；
* 集成了使用手册，方便查阅；
* 将所有模型的依赖放到根目录，安装依赖更加方便


# YoloAll安装
操作系统：Win10、Win7、Ubuntu16.04（其他操作系统没有测试过）  
python环境：python3.7  
cuda环境：cuda10.1（也可以不用cuda，使用cpu推理）  

### 下载YoloAll源码：
https://github.com/DL-Practise/YoloAll

### 解压YoloAll，进入到根目录
cd /path/to/yoloall
### 升级pip
python -m pip install --upgrade pip
### 安装依赖项
pip install -r requirements.txt


# YoloAll使用
### 启动YoloAll
python main_widget.py
		
### 界面简介
主要包括模型管理界面，配置界面，预测界面、消息界面和日志界面组成。
模型管理界面会将所有的模型以及子模型（例如YoloX包含s,m,l等子模型）
配置界面用于配置一些预测参数，例如置信度阈值，nms阈值，图片大小等
预测界面用于展现预测结果
消息界面展现YoloAll的升级信息，使用教程等
日志界面：展示一些关键的日志信息
![image](https://user-images.githubusercontent.com/16218143/143548998-e701af6b-af56-46ca-8712-26ed5965fdb7.png)

### 模型加载
刚启动YoloAll的时候，软件会去加载所有的模型，包括所有的子模型，因此需要等待一段时间，此时在日志界面会有相关的提示信息，需要等到日志界面出现“加载模型结束”。此时模型管理界面会出现所有加载的模型，表示模型的加载已经结束，可以进行测试了。
![image](https://user-images.githubusercontent.com/16218143/143549070-d420f63e-40d9-48a8-b855-53b394bda835.png)

### 下载预训练模型
由于预训练模型通常都很大，因此，下载的YoloAll中是不包含预训练模型的，需要通过鼠标点击模型管理界面中的子模型，例如点击YoloX下面的yolox_l模型，会弹出提示框，提示预训练模型未下载，然后会在提示框中展示预训练模型的下载链接（有可能有多个下载链接），选择一个网速不错的链接，从浏览器下载，下载完毕之后，根据提示框中的提示信息，放到对应的文件夹下面。
![image](https://user-images.githubusercontent.com/16218143/143549133-c6de1bb6-9ba3-4530-88b8-32981ecf8958.png)

### 预测
下载完预训练模型，并放到指定文件夹后，再次点击模型管理界面的模型，就会开始创建模型，并显示创建模型成功的界面  
![image](https://user-images.githubusercontent.com/16218143/143549185-4716bc4a-0d40-436c-b2b3-b97cc964f0f3.png)  
此时可以点击预测界面的Photo按钮，选择一张图片进行预测，也可以点击Video按键，选择一段视频文件进行预测，也可以点击Camera按钮，启动摄像头进行拍摄和预测。

### 修改预测参数
在预测过程中，如果需要进行CPU、GPU的切换，或者修改预测参数，例如修改图像分辨率，置信度阈值，nms阈值等，可以在配置界面进行修改的勾选或者修改，修改完后，点击保存按钮，软件会重新创建模型，等模型创建成功后，可以继续使用新的配置进行预测。  
![image](https://user-images.githubusercontent.com/16218143/143549367-2fc14ca1-3427-4dda-988d-d73abe40750d.png)  


### 预测信息
预测成功后，会在预测界面展示预测的信息，图中的方框即为预测出来的目标，红色的文字是预测的速度和FPS。大家可以切换不同的模型，进行预测结果以及耗时的对比。  
![image](https://user-images.githubusercontent.com/16218143/143549379-4f4281ef-b6e0-4953-b2b9-8118b1853751.png)  

		
# YoloAll演示视频(V2.0)  
后续补充。。。

