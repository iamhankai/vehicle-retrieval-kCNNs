# Vehicle Retrieval
Multi-CNN feature ensemble method. 

1. We have trained 4 GoogLeNet with different task, e.g, triplet loss (full image), vehicle_id softmax loss (full image), vehicle_id softmax loss (upper half image) and model_id softmax loss (lower half image). 
2. In inference stage, we concat the `pool5/7x7_s1` layer feature from the 4 GoogLeNet together. Finally, return the `k` nearest vehicles.

## Competition & Dataset
We used the method in Vehicle Retrieval task of [*The 3rd National Gradute Contest on Smart-CIty Technology and Creative Design, China*](http://www.smartcity-competition.com.cn/). We ranked **1st** and won the **special prize** in the final!

The Dataset used in Vehicle Retrieval task: 
[PKU VehicleID](http://www.pkuml.org/research/pku-vehicleid.html). Note: if you want to use the dataset, go to the website and ask for the download link.

## Platform
- CPU or GPU: CPU only
- OS: Windows x64
- DL tool: Caffe
- Compiler: VS2013

## Usage
1. Windows Caffe setup. Details in [Windows Caffe readme](https://github.com/BVLC/caffe/tree/windows).
2. Download or git clone the current project.
3. Copy or move `vs_vehicle_retrieval_kCNNs` folder into `caffe/windows` and add `vehicle_retrieval_kCNNs.vcxproj` project into Caffe solution in VS2013, compile it.
4. Modify `run.bat`, mainly set the path. Finally, run `run.bat` in cmd, you'll get a `xml` result file.

---

# 车辆精确检索
第三届全国研究生智慧城市技术与创意设计大赛车辆精确检索任务第一名，总决赛特等奖。

数据集：[PKU VehicleID](http://www.pkuml.org/research/pku-vehicleid.html)

## 方法
基于深度学习的多模型集成方法。

## 平台
CPU，Windows系统，Caffe，VS2013

## 使用
1. 下载、配置、编译Caffe官方windows版
2. 下载本工程
3. 将文件夹`vs_vehicle_retrieval_kCNNs`复制到Caffe/w`caffe/windows`目录下，并在vs中把`vehicle_retrieval_kCNNs.vcxproj`项目添加到Caffe解决方案下，编译生成可执行文件。
4. 修改`run.bat`中的路径，运行它，即可得到实验结果。


