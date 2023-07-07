# ImprovedSimpleHRNet_KeyPointDetector

이 프로젝트는 [Multi-person Human Pose Estimation with HRNet in PyTorch 의 ](https://arxiv.org/abs/1902.09212)https://arxiv.org/abs/1902.09212 비공식 구현이며, 완전히 같은 결과를 제공합니다.

offical Code : https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

또한 simple-HRNet의 구조와 같습니다. 즉, 설치와 실행 방법이 동일합니다.

simple-HRNet : https://github.com/stefanopini/simple-HRNet

YOLOX를 사용하여 기존보다 높은 성능을 보여줍니다. 물론, 기존의 YOLOv3, YOLOv5도 여전히 사용 가능합니다.

####YOLOv5

  
  ![HRNet_Yolov5](https://github.com/startedourmission/ImprovedSimpleHRNet_KeyPointDetector/assets/53049011/45a9c7a0-b611-4a6d-a4e4-abf9bf8ed9d5)


####YOLOX

  
![HRNet_Yolox](https://github.com/startedourmission/ImprovedSimpleHRNet_KeyPointDetector/assets/53049011/d814f2e6-3d5d-49fe-8658-10d7272f558e)


* * * 

# YOLOX 사용

먼저, 다른 YOLO를 사용할 때와 마찬가지로, models_/detector/ 아래 YOLO를 설치합니다.


https://github.com/Megvii-BaseDetection/YOLOX


그 다음, demo.py를 다음과 같이 수정합니다.


```
#model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth", yolo_version='v5', yolo_model_def='yolov5m', device=device)
model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth", yolo_version='x', device=device)
```
크기, pth 파일, device와 관계없이 yolo_version을 'x'로 하기만 하면 됩니다. X버전에서 yolo_model_def는 필요없습니다.
