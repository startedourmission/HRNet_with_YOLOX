import os
import time

import cv2
import numpy as np
import torch
from torch import tensor

from models_.detectors.yolox.yolox.data import ValTransform
from models_.detectors.yolox.yolox.exp import *
from models_.detectors.yolox.yolox.utils import *


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class YOLOX:
    def __init__(self,
                 model_def='',
                 model_folder='/home/titan/Jinwoo/yolox',
                 image_resolution=(640, 640),
                 conf_thres=0.1,
                 device=torch.device('cpu'),
                 ):

        self.model_def = model_def
        self.model_folder = model_folder
        self.image_resolution = image_resolution
        self.conf_thres = conf_thres
        self.device = device

        self.exp = get_exp('models_/detectors/yolox/exps/default/yolox_x.py', 'yolox_x')
        self.preproc = ValTransform(legacy=False)
        self.num_classes = self.exp.num_classes
        self.confthre = self.exp.test_conf
        self.nmsthre = self.exp.nmsthre

        # Set up model

            # load the pre-trained yolox in a pre-defined folder
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            # self.model = torch.hub.load('ultralytics/yolox', self.model_def, pretrained=True)
            # self.exp = get_exp(self.exp, 'yolox_x')
        self.model = self.exp.get_model()
        self.model = self.model.to(self.device)
        self.model.eval()  # Set in evaluation mode

    def predict_single(self, image, color_mode='BGR'):
        image = image.copy()


        if color_mode == 'BGR':
            # all YOLO models expect RGB
            # See https://github.com/ultralytics/yolov5/issues/9913#issuecomment-1290736061 and
            # https://github.com/ultralytics/yolov5/blob/8ca182613499c323a411f559b7b5ea072122c897/models/common.py#L662
            # image = image[..., ::-1]
            pass

        with torch.no_grad():
            img, _ = self.preproc(image, None, self.exp.test_size)
            img = torch.from_numpy(img).unsqueeze(0)
            img = img.cuda()
            ckpt = 'models_/detectors/yolox/yolox_x.pth'
            model = self.exp.get_model()
            if ckpt is not None:
                model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
            else:
                raise Exception("No checkpoint file specified.")

            # Use GPU if available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)
            # Put model in eval mode
            model.eval()
            ratio = min(self.image_resolution[0] / img.shape[0], self.image_resolution[1] / img.shape[1])
            output = model(img)
            output = postprocess(
                output, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )

            #detections = cxcywh2xyxy(output[0])
            detections = output[0]
            # print(detections)

            detections = detections[detections[:, 5] * detections[:, 4] >= self.conf_thres]
            detections = detections[detections[:, 6] == 0.]  # person
            detections[:, :4] = detections[:, :4] / ratio * 145
            detections[detections < 0] = 0

            # bcnt = 0
            # for box in detections:
            #     #box = box[:4] / ratio * 100
            #     x1 = int(round(box[0].item()))
            #     x2 = int(round(box[2].item()))
            #     y1 = int(round(box[1].item()))
            #     y2 = int(round(box[3].item()))
            #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
            #     bcnt += 1
            # cv2.putText(image, "bbox : " + str(bcnt), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            # cv2.imshow('img', image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        return detections

    def predict(self, images, color_mode='BGR'):
        raise NotImplementedError("Not currently supported.")
