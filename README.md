# ImprovedSimpleHRNet_KeyPointDetector

이 프로젝트는 [Multi-person Human Pose Estimation with HRNet in PyTorch 의 ](https://arxiv.org/abs/1902.09212)https://arxiv.org/abs/1902.09212 비공식 구현이며, 완전히 같은 결과를 제공합니다.

offical Code : https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

또한 simple-HRNet의 구조와 같습니다. 즉, 설치와 실행 방법이 동일합니다.

simple-HRNet : https://github.com/stefanopini/simple-HRNet

YOLOX를 사용하여 기존보다 높은 성능을 보여줍니다. 
기존의 YOLOv3, YOLOv5도 여전히 사용 가능합니다.

* * * 

#YOLOX 사용

demo.py를 다음과 같이 수정합니다.


```
27 #model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth", yolo_version='v5', yolo_model_def='yolov5m', device=device)
28 model = SimpleHRNet(32, 17, "./weights/pose_hrnet_w32_256x192.pth", yolo_version='x', device=device)
```
