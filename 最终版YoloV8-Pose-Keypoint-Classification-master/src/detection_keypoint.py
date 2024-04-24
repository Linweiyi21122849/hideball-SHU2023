import sys
import cv2
import numpy as np
from pydantic import BaseModel

import ultralytics
from ultralytics.engine.results import Results

# Define keypoint
# 关键点名称索引
class GetKeypoint(BaseModel): 
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3 
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5 #*************
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11 #*************
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16

class DetectKeypoint:
    # 构造函数
    def __init__(self, yolov8_model='yolov8n-pose'): 
        self.yolov8_model = yolov8_model
        # get_ketpoint相当于索引数组
        self.get_keypoint = GetKeypoint()
        self.__load_model()

    #确保加载的是姿势检测模型
    def __load_model(self):
        if not self.yolov8_model.split('-')[-1] == 'pose':
            sys.exit('Model not yolov8 pose')
        self.model = ultralytics.YOLO(model=self.yolov8_model)

        # extract function keypoint 
        #表示函数将返回一个列表
    def extract_keypoint(self, keypoint: np.ndarray) -> list:
        # nose
        nose_x, nose_y = keypoint[self.get_keypoint.NOSE]
        # eye
        left_eye_x, left_eye_y = keypoint[self.get_keypoint.LEFT_EYE]
        right_eye_x, right_eye_y = keypoint[self.get_keypoint.RIGHT_EYE]
        # ear
        left_ear_x, left_ear_y = keypoint[self.get_keypoint.LEFT_EAR]
        right_ear_x, right_ear_y = keypoint[self.get_keypoint.RIGHT_EAR]
        # shoulder
        left_shoulder_x, left_shoulder_y = keypoint[self.get_keypoint.LEFT_SHOULDER]
        right_shoulder_x, right_shoulder_y = keypoint[self.get_keypoint.RIGHT_SHOULDER]
        # elbow
        left_elbow_x, left_elbow_y = keypoint[self.get_keypoint.LEFT_ELBOW]
        right_elbow_x, right_elbow_y = keypoint[self.get_keypoint.RIGHT_ELBOW]
        # wrist
        left_wrist_x, left_wrist_y = keypoint[self.get_keypoint.LEFT_WRIST]
        right_wrist_x, right_wrist_y = keypoint[self.get_keypoint.RIGHT_WRIST]
        # hip
        left_hip_x, left_hip_y = keypoint[self.get_keypoint.LEFT_HIP]
        right_hip_x, right_hip_y = keypoint[self.get_keypoint.RIGHT_HIP]
        # knee
        left_knee_x, left_knee_y = keypoint[self.get_keypoint.LEFT_KNEE]
        right_knee_x, right_knee_y = keypoint[self.get_keypoint.RIGHT_KNEE]
        # ankle
        left_ankle_x, left_ankle_y = keypoint[self.get_keypoint.LEFT_ANKLE]
        right_ankle_x, right_ankle_y = keypoint[self.get_keypoint.RIGHT_ANKLE]
        
        return [
            nose_x, nose_y, left_eye_x, left_eye_y, right_eye_x, right_eye_y,
            left_ear_x, left_ear_y, right_ear_x, right_ear_y, left_shoulder_x, left_shoulder_y,
            right_shoulder_x, right_shoulder_y, left_elbow_x, left_elbow_y, right_elbow_x, right_elbow_y,
            left_wrist_x, left_wrist_y, right_wrist_x, right_wrist_y, left_hip_x, left_hip_y,
            right_hip_x, right_hip_y, left_knee_x, left_knee_y, right_knee_x, right_knee_y,        
            left_ankle_x, left_ankle_y,right_ankle_x, right_ankle_y
        ]
    
    #接受一个 Results作为参数，Results中包含预测的关键点,放回一维的关键点信息
    def get_xy_keypoint(self, results: Results) -> list:
        result_keypoint = results.keypoints.xyn.cpu().numpy()[0]
        keypoint_data = self.extract_keypoint(result_keypoint)
        return keypoint_data
    
    #允许将 DetectKeypoint 对象作为可调用对象使用。它接受一个图像数组 image 作为参数，并使用 YOLO 模型对图像进行预测。
    def __call__(self, image: np.array) -> Results:
        results = self.model.predict(image, save=False)[0]
        return results

