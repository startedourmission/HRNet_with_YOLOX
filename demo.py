import os

import cv2
from SimpleHRNet import SimpleHRNet
from misc.visualization import joints_dict
from tqdm import tqdm
import torch


def cv_joints(image, outputs):

    for output in outputs:
        for joint in output:
            x, y, prob = joint
            cv2.circle(image, (int(y), int(x)), 2, (0, 0, 255), -1)

    return image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#DATA_PATH = '/home/titan/Jinwoo/DatasetTools/dataset/TurbulentFlow_Capture/'

# VIDEO_PATH = '/home/titan/Jinwoo/DatasetTools/dataset/NYPD_Video/'
DATA_PATH = '/home/titan/Jinwoo/DatasetTools/dataset/NYPD_Capture/'
DATA_FOLDER = 'Cap_3_1740_11_NYPD_Crowd_Behavior_Training_Video/'
IMG_NAME = 'frame0.jpg'

img = DATA_PATH + DATA_FOLDER + IMG_NAME

#model = SimpleHRNet(48, 17, "./weights/pose_hrnet_w48_384x288.pth", yolo_version='v5', yolo_model_def='yolov5m', device=device)
model = SimpleHRNet(32, 17, "./weights/pose_hrnet_w32_256x192.pth", yolo_version='x', device=device)

for i in tqdm(range(0, len(os.listdir(DATA_PATH + DATA_FOLDER)))):
    image = cv2.imread(DATA_PATH + DATA_FOLDER + 'frame' + str(i) + '.jpg', cv2.IMREAD_COLOR)

    joints, boxes = model.predict(image)
    cv_joints(image, joints)

    # bcnt = 0
    # for box in boxes:
    #     cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
    #     bcnt += 1
    #     #print(box)
    #     #break;
    # cv2.putText(image, "bbox : " + str(bcnt), (image.shape[1] - 130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    res = image
    cv2.imwrite('output/res' + str(i) + '.jpg', res)

