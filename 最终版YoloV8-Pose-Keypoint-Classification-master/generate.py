import cv2
from PIL import Image
from pydantic import BaseModel
import ultralytics
import os
import glob
     
#设置一个变量 dataset_root，表示数据集的根目录路径
# dataset_root = 'datasets\Stand_Squat' 
dataset_root = 'datasets\Billet_Handsup' 

#使用 os.listdir 函数获取 dataset_root 路径下的所有文件夹和文件的名称，并将它们存储在列表 pose_list 中
pose_list = os.listdir(dataset_root)
print(pose_list)

model = ultralytics.YOLO(model='yolov8x-pose.pt')
# 创建一个空列表 dataset_csv，用于存储最终的数据集
dataset_csv = []
class GetKeypoint(BaseModel):
    NOSE:           int = 0
    LEFT_EYE:       int = 1
    RIGHT_EYE:      int = 2
    LEFT_EAR:       int = 3
    RIGHT_EAR:      int = 4
    LEFT_SHOULDER:  int = 5
    RIGHT_SHOULDER: int = 6
    LEFT_ELBOW:     int = 7
    RIGHT_ELBOW:    int = 8
    LEFT_WRIST:     int = 9
    RIGHT_WRIST:    int = 10
    LEFT_HIP:       int = 11
    RIGHT_HIP:      int = 12
    LEFT_KNEE:      int = 13
    RIGHT_KNEE:     int = 14
    LEFT_ANKLE:     int = 15
    RIGHT_ANKLE:    int = 16
get_keypoint = GetKeypoint()
def extract_keypoint(keypoint):
    # nose
    nose_x, nose_y = keypoint[get_keypoint.NOSE]
    # eye
    left_eye_x, left_eye_y = keypoint[get_keypoint.LEFT_EYE]
    right_eye_x, right_eye_y = keypoint[get_keypoint.RIGHT_EYE]
    # ear
    left_ear_x, left_ear_y = keypoint[get_keypoint.LEFT_EAR]
    right_ear_x, right_ear_y = keypoint[get_keypoint.RIGHT_EAR]
    # shoulder
    left_shoulder_x, left_shoulder_y = keypoint[get_keypoint.LEFT_SHOULDER]
    right_shoulder_x, right_shoulder_y = keypoint[get_keypoint.RIGHT_SHOULDER]
    # elbow
    left_elbow_x, left_elbow_y = keypoint[get_keypoint.LEFT_ELBOW]
    right_elbow_x, right_elbow_y = keypoint[get_keypoint.RIGHT_ELBOW]
    # wrist
    left_wrist_x, left_wrist_y = keypoint[get_keypoint.LEFT_WRIST]
    right_wrist_x, right_wrist_y = keypoint[get_keypoint.RIGHT_WRIST]
    # hip
    left_hip_x, left_hip_y = keypoint[get_keypoint.LEFT_HIP]
    right_hip_x, right_hip_y = keypoint[get_keypoint.RIGHT_HIP]
    # knee
    left_knee_x, left_knee_y = keypoint[get_keypoint.LEFT_KNEE]
    right_knee_x, right_knee_y = keypoint[get_keypoint.RIGHT_KNEE]
    # ankle
    left_ankle_x, left_ankle_y = keypoint[get_keypoint.LEFT_ANKLE]
    right_ankle_x, right_ankle_y = keypoint[get_keypoint.RIGHT_ANKLE]
    return [
        nose_x, nose_y,
        left_eye_x, left_eye_y,
        right_eye_x, right_eye_y,
        left_ear_x, left_ear_y,
        right_ear_x, right_ear_y,
        left_shoulder_x, left_shoulder_y,
        right_shoulder_x, right_shoulder_y,
        left_elbow_x, left_elbow_y,
        right_elbow_x, right_elbow_y,
        left_wrist_x, left_wrist_y,
        right_wrist_x, right_wrist_y,
        left_hip_x, left_hip_y,
        right_hip_x, right_hip_y,
        left_knee_x, left_knee_y,
        right_knee_x, right_knee_y,        
        left_ankle_x, left_ankle_y,
        right_ankle_x, right_ankle_y
    ]
for pose in pose_list:
    image_path_list = glob.glob(f'{dataset_root}/{pose}/*.[jJ][pP][gG]')
    image_path_list += glob.glob(f'{dataset_root}/{pose}/*.[jJ][pP][eE][gG]')
    image_path_list += glob.glob(f'{dataset_root}/{pose}/*.[pP][nN][gG]')
    for image_path in image_path_list:
        # get image_name
        image_name = image_path.split('/')[-1]
        # read numpy image
        image = cv2.imread(image_path)
        # get height width image
        height, width = image.shape[:2]    
        # detect pose using yolov8-pose
        results = model.predict(image, save=False)[0]
        results_keypoint = results.keypoints.xyn.cpu().numpy()
        for result_keypoint in results_keypoint:
            if len(result_keypoint) == 17:
                keypoint_list = extract_keypoint(result_keypoint)
                # inset image_name, labe] in index 0,1
                keypoint_list.insert(0, image_name)
                keypoint_list.insert(1, pose)
                dataset_csv.append(keypoint_list)
        # break
    # break
import csv
# write csv
# 定义每条记录的每一列的名称
header = [
    'image_name',       # 0
    'label',            # 1
    'nose_x',           # 2
    'nose_y',           # 3
    'left_eye_x',       # 4
    'left_eye_y',       # 5
    'right_eye_x',      # 6
    'right_eye_y',      # 7
    'left_ear_x',       # 8     ***********
    'left_ear_y',       # 9
    'right_ear_x',      # 10
    'right_ear_y',      # 11
    'left_shoulder_x',  # 12
    'left_shoulder_y',  # 13
    'right_shoulder_x', # 14
    'right_shoulder_y', # 15
    'left_elbow_x',     # 16
    'left_elbow_y',     # 17
    'right_elbow_x',    # 18
    'right_elbow_y',    # 19
    'left_wrist_x',     # 20
    'left_wrist_y',     # 21
    'right_wrist_x',    # 22
    'right_wrist_y',    # 23   ***********
    'left_hip_x',       # 24   ***********
    'left_hip_y',       # 25
    'right_hip_x',      # 26
    'right_hip_y',      # 27
    'left_knee_x',      # 28
    'left_knee_y',      # 29
    'right_knee_x',     # 30
    'right_knee_y',     # 31
    'left_ankle_x',     # 32
    'left_ankle_y',     # 33
    'right_ankle_x',    # 34
    'right_ankle_y'     # 35
]

# 指定 newline=''  指定了行结束符为空
#将数据写入到名为yoga_pose_keypoint.csv的CSV文件中
# with open('datasets\Legspose_keypoint.csv', 'w', encoding='UTF8', newline='') as f:
with open('datasets\Handspose_keypoint.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(header)
    # write multiple rows
    writer.writerows(dataset_csv)