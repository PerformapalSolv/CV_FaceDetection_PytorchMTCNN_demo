[本项目链接](https://github.com/PerformapalSolv/CV_FaceDetection_PytorchMTCNN_demo)

https://github.com/PerformapalSolv/CV_FaceDetection_PytorchMTCNN_demo

# 人脸检测小作业

[TOC]

项目使用：直接使用Pytorch-MTCNN-Mychange/Pytorch-MTCNN-Origin中已经训练好的模型

在文件夹下运行:

```python
python infer_camera.py
或：
infer_path.py
```

如需自行训练模型:

下载dataset文件夹：链接：https://pan.baidu.com/s/1ClZdE-9XK1rZ4YFSJgkpfg?pwd=ltks    提取码：ltks 

接着:

> ```shell
> cd dataset
> python ChangeDataset.py   
> ```
>
> - `cd train_PNet` 切换到`train_PNet`文件夹
> - `python3 generate_PNet_data.py` 首先需要生成PNet模型训练所需要的图像数据
> - `python3 train_PNet.py` 开始训练PNet模型
>
> - `cd train_RNet` 切换到`train_RNet`文件夹
> - `python3 generate_RNet_data.py` 使用上一步训练好的PNet模型生成RNet训练所需的图像数据
> - `python3 train_RNet.py` 开始训练RNet模型
>
> - `cd train_ONet` 切换到`train_ONet`文件夹
> - `python3 generate_ONet_data.py` 使用上两部步训练好的PNet模型和RNet模型生成ONet训练所需的图像数据
> - `python3 train_ONet.py` 开始训练ONet模型

## Opencv级联分类器

>  OpenCV级联分类器的原理主要基于Viola和Jones在2001年提出的一种高效的特征检测方法。它通过使用一系列简单的特征来检测目标对象,并利用AdaBoost算法训练出一个强分类器。该方法主要包括以下几个关键步骤: 
>
> 1. **积分图像(Integral Image)** 为了快速计算图像的任意矩形区域的像素值之和,引入了积分图像的概念。积分图像可以通过简单的递推关系式快速生成。 
> 2. **Haar-like特征** 使用Haar-like特征来描述目标对象的特征,这些矩形特征通过加权求和计算得到。Haar-like特征对于垂直、水平和对角边缘等较为敏感。 
> 3. **AdaBoost算法** AdaBoost算法用于从大量的Haar-like特征中选择一小部分有效特征,并将这些弱分类器线性组合成一个强分类器。每个弱分类器根据单个特征的分类结果赋予加权系数。
> 4.  **级联结构** 为了提高检测速度,在训练时构建一个由多个加权强分类器构成的级联结构。待检测的窗口区域必须通过所有级别的分类器才能被判定为目标对象。大部分负样本在初级阶段就会被剔除,从而加快了检测速度。 
> 5. **滑动窗口检测** 利用滑动窗口的方式在图像上进行目标检测。窗口在图像四周和不同尺度扫描,对每个窗口区域利用级联分类器进行分类。 总的来说,级联分类器通过简单高效的特征、AdaBoost训练和级联结构,实现了快速且可靠的目标检测。它广泛应用于人脸、行人、车辆等目标的实时检测领域。

在下载Opencv-python包时，级联分类器模型已经自动下载

```python
C:/anaconda3/envs/pytorch/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml
```

如以下代码:

```python
#导入cv模块
import cv2 as cv
#检测函数
def face_detect_demo():
    gary = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier('C:/anaconda3/envs/pytorch/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    face = face_detect.detectMultiScale(gary,1.01,5,0,(100,100),(300,300))
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('result',img)

#读取图像
img = cv.imread('test.jpg')
#检测函数
face_detect_demo()
#等待
while True:
    if ord('q') == cv.waitKey(0):
        break
#释放内存
cv.destroyAllWindows()

```



## Pytorch-MTCNN-Origin：直接使用MTCNN的训练过程

**最大的困难在于数据集的选择，我选择[Deep Convolutional Network Cascade for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) 人脸关键点数据集、[WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)人脸 数据集**

这两个都是相当大的数据集，直接训练不可行，所以要对数据集再进行选择。最终，得到可以训练的数据集。

[原项目作者](https://github.com/yeyupiaoling/Pytorch-MTCNN)对WIDER_FACE集数据集进行精简，已经得到了12000张图片的数据集——但在此基础上，以我当前算力还是很耗费时间——对此，我进行进一步裁剪，只取其中人脸数<=2的图片，最后得到约6820张图片，实验得以进行，并最终浮现训练过程，训练出自己的模型

MTCNN网络：

[论文链接](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)

![image-20240407203616293](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407203616293.png)

配置：2*P100-PCIE-16GB

<img src="https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407194653094.png" alt="image-20240407194653094" style="zoom: 67%;" />



### Pnet

#### generate_Pnet_data

![image-20240407162455478](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407162455478.png)

#### train_Pnet

![image-20240407163949530](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407163949530.png)

一共训练30个周期，每次384个批次，学习率为1e-3

![image-20240407164056338](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407164056338.png)

### Rnet

#### Generate_Rnet

![image-20240407183839914](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407183839914.png)

#### Train_Rnet

![image-20240407183955539](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407183955539.png)

```
batch_size = 384
learning_rate = 1e-3
epoch_num = 22
```

![image-20240407184642606](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407184642606.png)

### Onet

#### Generate_Onet

![image-20240407191753799](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407191753799.png)

#### train_Onet

![image-20240407192053774](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407192053774.png)

![image-20240407192108657](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407192108657.png)

![image-20240407192527716](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407192527716.png)



### 实验结果：

<img src="https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408022139955.png" alt="image-20240408022139955" style="zoom:67%;" />



## Pytorch-MTCNN-Mychange:对网络进行修改，删除Landmask部分，简化模型

**为了简化实验代码，调整网络结构，删去PNet/RNet/ONet中的Landmask部分，减少关键点训练的部分**

结果：模型训练过程快了大概40%左右，很快能训练出结果。

### Pnet

#### Generate_Pnet_data

![image-20240407205551773](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407205551773.png)

#### Train_Pnet

![image-20240407210553593](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407210553593.png)

![image-20240407210606847](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240407210606847.png)

### RNet

#### Generate_Rnet_data

![image-20240408010353332](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408010353332.png)

#### Train_Rnet

![image-20240408010642717](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408010642717.png)

![image-20240408011204426](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408011204426.png)

### ONet

#### Generate_Onet_data

![image-20240408013948236](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408013948236.png)

#### Train_Onet

![image-20240408014125545](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408014125545.png)

![image-20240408014345699](https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408014345699.png)

### 模型效果

**图片**

<img src="https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408015625725.png" alt="image-20240408015625725" style="zoom:67%;" />

**视频:**

<img src="https://cdn.jsdelivr.net/gh/PerformapalSolv/githubChartBed@main/img/image-20240408015556984.png" alt="image-20240408015556984" style="zoom:67%;" />
