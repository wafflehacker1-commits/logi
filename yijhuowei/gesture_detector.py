# modules/vision/gesture_detector.py

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

class GestureDetector:
    """
    使用 MediaPipe Tasks 偵測手部和手勢
    """
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        初始化手勢偵測器
        
        Args:
            confidence_threshold: 偵測信心度閾值 (0.0-1.0)
        """
        self.confidence_threshold = confidence_threshold
        
        # 下載模型（如果需要）
        try:
            # 使用 MediaPipe Tasks 的手部追蹤
            base_options = python.BaseOptions(
                
                model_asset_path='models/hand_landmarker.task'
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=confidence_threshold,
                min_hand_presence_confidence=confidence_threshold,
                min_tracking_confidence=confidence_threshold
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"⚠️ Could not load hand_landmarker.task model: {e}")
            print("Using fallback OpenCV hand detection...")
            self.landmarker = None
        
        print("✅ GestureDetector initialized successfully")
        
    def detect_hands(self, frame: np.ndarray) -> Dict:
        """
        偵測影像中的手部
        
        Args:
            frame: OpenCV 影像幀 (BGR 格式)
            
        Returns:
            包含偵測結果的字典
        """
        h, w, c = frame.shape
        
        if self.landmarker is None:
            return self._detect_hands_fallback(frame)
        
        try:
            # 轉換 BGR 到 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 使用 MediaPipe 偵測
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = self.landmarker.detect(mp_image)
            
            # 準備返回數據
            hand_data = {
                'detected': len(detection_result.hand_landmarks) > 0,
                'hands': [],
                'handedness': []
            }
            
            for hand_landmarks, handedness in zip(
                detection_result.hand_landmarks,
                detection_result.handedness
            ):
                # 提取關鍵點
                landmarks = self._extract_landmarks(hand_landmarks)
                hand_data['hands'].append(landmarks)
                hand_data['handedness'].append(handedness[0].category_name)
            
            return hand_data
            
        except Exception as e:
            print(f"Detection error: {e}")
            return self._detect_hands_fallback(frame)
    
    def _detect_hands_fallback(self, frame: np.ndarray) -> Dict:
        """
        備用手部偵測方法（當模型不可用時）
        使用膚色檢測
        """
        h, w, c = frame.shape
        
        # 轉換到 HSV 色彩空間
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 定義膚色範圍（HSV）
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # 形態學操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 尋找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hands = []
        handedness = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # 最小面積
                # 生成假的手部關鍵點（簡化版本）
                x, y, w_c, h_c = cv2.boundingRect(contour)
                landmarks = self._generate_dummy_landmarks(x, y, w_c, h_c, frame.shape)
                hands.append(landmarks)
                handedness.append("Right")
        
        return {
            'detected': len(hands) > 0,
            'hands': hands,
            'handedness': handedness
        }
    
    def _generate_dummy_landmarks(self, x: int, y: int, w: int, h: int, frame_shape: Tuple) -> List:
        """
        生成虛擬的 21 個關鍵點（用於備用方法）
        """
        frame_h, frame_w, _ = frame_shape
        landmarks = []
        
        # 根據邊界框生成 21 個點
        for i in range(21):
            px = (x + w * (i % 5) / 5) / frame_w
            py = (y + h * (i // 5) / 5) / frame_h
            landmarks.append((px, py, 0.5))
        
        return landmarks
    
    def _extract_landmarks(self, hand_landmarks) -> List[Tuple[float, float, float]]:
        """
        從 MediaPipe 手部模型提取 21 個關鍵點
        
        Returns:
            21 個 (x, y, z) 座標的列表
        """
        landmarks = []
        for landmark in hand_landmarks:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks
    
    def recognize_gesture(self, landmarks: List[Tuple[float, float, float]]) -> str:
        """
        根據手部關鍵點識別手勢
        
        Args:
            landmarks: 21 個手部關鍵點
            
        Returns:
            手勢名稱 ('open_hand', 'fist', 'point', 'thumbs_up', 'peace', 'unknown')
        """
        if not landmarks or len(landmarks) < 21:
            return 'unknown'
        
        # 計算手指伸展狀態
        fingers_extended = self._count_extended_fingers(landmarks)
        
        # 基於伸展的手指數量識別手勢
        if fingers_extended == 5:
            return 'open_hand'
        elif fingers_extended == 0:
            return 'fist'
        elif fingers_extended == 1:
            return 'point'
        elif fingers_extended == 2:
            return 'peace'
        else:
            return 'other'
    
    def _count_extended_fingers(self, landmarks: List[Tuple[float, float, float]]) -> int:
        """
        計算伸展的手指數量
        
        MediaPipe 手部模型有 21 個關鍵點：
        - 0: 手腕
        - 1-4: 大拇指
        - 5-8: 食指
        - 9-12: 中指
        - 13-16: 無名指
        - 17-20: 小指
        """
        extended = 0
        
        # 大拇指 (比較 x 座標)
        if landmarks[4][0] < landmarks[3][0]:
            extended += 1
        
        # 其他手指 (比較 y 座標)
        for finger_tip, finger_pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if landmarks[finger_tip][1] < landmarks[finger_pip][1]:
                extended += 1
        
        return extended
    
    def draw_landmarks(self, frame: np.ndarray, hand_data: Dict) -> np.ndarray:
        """
        在影像上繪製手部關鍵點和連接線
        
        Args:
            frame: 原始影像幀
            hand_data: 偵測結果
            
        Returns:
            帶有繪製的影像幀
        """
        output_frame = frame.copy()
        
        if not hand_data['hands']:
            return output_frame
        
        h, w, c = output_frame.shape
        
        for i, hand_landmarks_list in enumerate(hand_data['hands']):
            # 繪製每個關鍵點
            for landmark in hand_landmarks_list:
                x = int(landmark[0] * w)
                y = int(landmark[1] * h)
                cv2.circle(output_frame, (x, y), 5, (0, 255, 0), -1)
            
            # 繪製連接線 (Hand connections from MediaPipe)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
                (5, 9), (9, 13), (13, 17)  # Palm
            ]
            
            for start, end in connections:
                x1 = int(hand_landmarks_list[start][0] * w)
                y1 = int(hand_landmarks_list[start][1] * h)
                x2 = int(hand_landmarks_list[end][0] * w)
                y2 = int(hand_landmarks_list[end][1] * h)
                cv2.line(output_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # 顯示手勢和慣用手
            gesture = self.recognize_gesture(hand_landmarks_list)
            handedness_label = hand_data['handedness'][i] if i < len(hand_data['handedness']) else "Unknown"
            
            # 找到手腕關鍵點的像素座標作為文本位置
            wrist_x = int(hand_landmarks_list[0][0] * w)
            wrist_y = int(hand_landmarks_list[0][1] * h) - 20
            
            cv2.putText(
                output_frame,
                f"{handedness_label}: {gesture}",
                (wrist_x, wrist_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
                cv2.LINE_AA
            )
        
        return output_frame
    
    def release(self):
        """釋放資源"""
        if self.landmarker:
            self.landmarker = None
