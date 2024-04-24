import cv2
import numpy as np
from PIL import Image
from src.detection_keypoint import DetectKeypoint
from src.Hands_keypoint import HandsClassification
from src.Legs_keypoint import LegsClassification
from src.YeahGestureCounter import YeahGestureCounter

detection_keypoint = DetectKeypoint()
yeah_counter = YeahGestureCounter()
Hand_keypoint = HandsClassification('models\Handspose_classification.pt')
Leg_keypoint = LegsClassification('models\Legspose_classification.pt')

frame = cv2.imread('images/888.jpg')

results = detection_keypoint(frame)
results_keypoint = detection_keypoint.get_xy_keypoint(results)#得到人体姿态关键点坐标数组

tmp1 = results_keypoint[10:22]  # 提取第6-21个关键点，表示提取上肢
handslable, hand_confidence = Hand_keypoint(tmp1)  # 进行手部标签预测
tmp2 = results_keypoint[22:]  # 从第22个关键点开始提取，表示提取下肢
legslable, leg_confidence = Leg_keypoint(tmp2)  # 进行腿部标签预测
#可视化
height, width = frame.shape[:2]

annotated_frame = results.plot(boxes=False) #生成一个可视化关键点图像

x_min, y_min, x_max, y_max = results.boxes.xyxy[0].cpu().numpy()

# frame2 = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
# Image._show(Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)))
cnt = yeah_counter.get_cnt(frame)  # 进行额外的判断
print("cnt :" +str(cnt))

annotated_frame = cv2.rectangle(
                annotated_frame, 
                (int(x_min), int(y_min)),(int(x_max), int(y_max)), 
                (0,0,255), 2
            )

hand_text = f'{handslable.upper()}: {hand_confidence:.2f}'
leg_text = f'{legslable.upper()}: {leg_confidence:.2f}'
combined_text = f'{hand_text} {leg_text}'  # 组合手部和腿部的标签和置信度

text_size, _ = cv2.getTextSize(combined_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
text_width, text_height = text_size
# 绘制矩形背景
rect_left = int(x_min)
rect_top = int(y_min) - text_height - 20
rect_right = int(x_min) + text_width + 20
rect_bottom = int(y_min)
annotated_frame = cv2.rectangle(
    annotated_frame,
    (rect_left, rect_top),
    (rect_right, rect_bottom),
    (0, 0, 255),
    cv2.FILLED
)
cv2.putText(annotated_frame,
            combined_text,
            (int(x_min), int(y_min)-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA
            )

print(f'Keypoint classification: {combined_text}')
Image._show(Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)))


