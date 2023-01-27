
# **GLMDriveNet: Global-local Multimodal Fusion Driving Behavior Classification Network**

PyTorch implementation of the paper "GLMDriveNet: Global-local Multimodal Fusion Driving Behavior Classification Network"



## **Changelog**



- [2023-01-28] Release the initial code for GLMDriveNet.



## **一、Dataset processing**



#### **UAH-DriveSet dataset**

1.Origin UAH-DriveSet : This dataset is captured by DriveSafe, a driving monitoring application.The application is run by 6 different drivers and vehicles, performing 3 different driving behaviors (normal, drowsy and aggressive) on two types of roads (motorway and secondary road), resulting in more than 500 minutes of naturalistic driving with its associated raw data and processed semantic information, together with the video recordings of the trips. The UAH-DriveSet is available at: http://www.robesafe.com/personal/eduardo.romera/uah-driveset.

2.Processes UAH-DriveSet: First of all, since the UAH-DriveSet captures the roadside video during driving, we extract the last frame of every second from the video. Secondly, we interpolate the vehicle speed in RAW_GPS file and expand it to 1260 data every second. Then merge the data of *x* seconds and the first four seconds into a txt file named *x.txt*, which corresponds to the last frame image *x.jpg* of x seconds. Please refer to: [Baidu Cloud link](https://pan.baidu.com/s/1BOK_4rewfofSY79V82muEg?pwd=44sX)




## **二、Quick Start**

#### 1.Environment configuration: Clone repo and install requirements.txt in a Python>=3.6.0 environment, including PyTorch>=1.7.

```
git clone https://github.com/liuwenzhuo1/GLMDrivenet.git
pip install -r requirements.txt  # install
```

#### 2.train

```
python main.py --mode train
```

#### 3.test

```
python main.py --mode test
```

#### 4.result

Performance comparison with other driving behavior classification methods on experimental data of all roads. The Acc, Pre and Rec represent the accuracy, precision and recall. The "-" means that it is not indicated in the method.

![](resultall.PNG)



Performance comparison with other driving behavior classification methods on experimental data of motorway road.

![](resultmotor.PNG)



Performance comparison with other driving behavior classification methods on experimental data of secondary road.

![](resultsecond.PNG)

## **Contribute**


Thanks to [Yan Gong](https://github.com/gongyan1) and [Wenzhuo Liu](https://github.com/liuwenzhuo1) for their contributions to this code base.
