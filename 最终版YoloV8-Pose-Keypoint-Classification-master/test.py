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

video_path = 0 #使用默认的摄像头设备
cap = cv2.VideoCapture(video_path) #通过 cv2.VideoCapture 打开视频流s
# 设置图像宽度和高度
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)
# 设置期望的帧率
desired_fps = 30
cap.set(cv2.CAP_PROP_FPS, desired_fps)

while cap.isOpened():
    success, frame = cap.read() #frame 变量保存了读取到的视频帧。
    results = detection_keypoint(frame)#使=用predict函数预测结果

    if len(results.boxes.data) > 0:  # 检查是否有检测结果
        results_keypoint = detection_keypoint.get_xy_keypoint(results)#得到人体姿态关键点坐标数组

        tmp1 = results_keypoint[10:22]  # 提取第6-21个关键点，表示提取上肢
        handslable, hand_confidence = Hand_keypoint(tmp1)  # 进行手部标签预测
        tmp2 = results_keypoint[22:]  # 从第22个关键点开始提取，表示提取下肢
        legslable, leg_confidence = Leg_keypoint(tmp2)  # 进行腿部标签预测
        #可视化
        height, width = frame.shape[:2]

        annotated_frame = results.plot(boxes=False) #生成一个可视化关键点图像

        x_min, y_min, x_max, y_max = results.boxes.xyxy[0].cpu().numpy()
        annotated_frame = cv2.rectangle(
                        annotated_frame, 
                        (int(x_min), int(y_min)),(int(x_max), int(y_max)), 
                        (0,0,255), 2
                    )
        
        combined_text = ''  # 初始化组合的标签和置信度

        if handslable == 'Billet' and hand_confidence >= 0.8:
            hand_text = f'{handslable.upper()}: {hand_confidence:.2f}'
            # 裁剪frame得到frame2，该帧因此只包含一个人
            frame2 = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
            cnt = yeah_counter.get_cnt(frame2)  # 进行额外的判断
            if cnt == 1:
                combined_text += hand_text + ' '
            # hand_text = f'{handslable.upper()}: {hand_confidence:.2f}'
            # combined_text += hand_text + ' '

        if handslable == 'Handsup' and hand_confidence >= 0.8:
            hand_text = f'{handslable.upper()}: {hand_confidence:.2f}'
            combined_text += hand_text + ' '

        if legslable == 'Squat' and leg_confidence >= 0.8:
            leg_text = f'{legslable.upper()}: {leg_confidence:.2f}'
            combined_text += leg_text + ' '

        text_size, _ = cv2.getTextSize(combined_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
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
                    1.2, (255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                    )

        print(f'Keypoint classification: {combined_text}')
        # 手动调整窗口大小以适应图像尺寸
        cv2.waitKey(1)  # 等待一小段时间以确保窗口大小调整生效
        cv2.imshow("YOLOv8 pose inference", annotated_frame)
    else:
        # 手动调整窗口大小以适应图像尺寸
        cv2.waitKey(1)  # 等待一小段时间以确保窗口大小调整生效
        cv2.imshow("YOLOv8 pose inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# 释放视频流资源，并关闭显示窗口
cap.release()
cv2.destroyAllWindows()