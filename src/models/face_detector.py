"""
Модуль детекции лиц
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image
import os

class FaceDetector:
    """Класс для детекции лиц на изображении"""
    
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        
        if not use_mock:
            try:
                from mtcnn import MTCNN
                self.detector = MTCNN()
                self.detection_method = "mtcnn"
            except ImportError:
                print("MTCNN не установлен. Используем OpenCV Haar cascades.")
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self.detection_method = "haar"
        else:
            self.detection_method = "mock"
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Детектирование лиц на изображении
        
        Args:
            image: Изображение в формате BGR (OpenCV)
        
        Returns:
            Список словарей с информацией о найденных лицах:
            - 'bbox': [x, y, width, height]
            - 'confidence': уверенность детекции
            - 'landmarks': ключевые точки лица (если доступно)
        """
        if self.use_mock:
            return self._mock_detect(image)
        
        if self.detection_method == "mtcnn":
            return self._detect_mtcnn(image)
        elif self.detection_method == "haar":
            return self._detect_haar(image)
        else:
            return self._mock_detect(image)
    
    def _detect_mtcnn(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Детекция с использованием MTCNN"""
        # Конвертируем BGR в RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Детекция лиц
        detections = self.detector.detect_faces(rgb_image)
        
        results = []
        for det in detections:
            if det['confidence'] < 0.9:  # Порог уверенности
                continue
            
            x, y, width, height = det['box']
            # MTCNN возвращает [x, y, width, height]
            bbox = [int(x), int(y), int(width), int(height)]
            
            results.append({
                'bbox': bbox,
                'confidence': float(det['confidence']),
                'landmarks': det.get('keypoints', {})
            })
        
        return results
    
    def _detect_haar(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Детекция с использованием Haar cascades"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Детекция лиц
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            results.append({
                'bbox': [int(x), int(y), int(w), int(h)],
                'confidence': 0.95,  # Haar не возвращает confidence
                'landmarks': {}  # Haar не возвращает landmarks
            })
        
        return results
    
    def _mock_detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Mock детекция для демо"""
        height, width = image.shape[:2]
        
        # Создаём mock детекцию в центре изображения
        bbox_size = min(height, width) // 3
        x = (width - bbox_size) // 2
        y = (height - bbox_size) // 2
        
        return [{
            'bbox': [x, y, bbox_size, bbox_size],
            'confidence': 0.95,
            'landmarks': {
                'left_eye': (x + bbox_size // 3, y + bbox_size // 3),
                'right_eye': (x + 2 * bbox_size // 3, y + bbox_size // 3),
                'nose': (x + bbox_size // 2, y + bbox_size // 2),
                'mouth_left': (x + bbox_size // 3, y + 2 * bbox_size // 3),
                'mouth_right': (x + 2 * bbox_size // 3, y + 2 * bbox_size // 3)
            }
        }]
    
    def extract_face(self, image: np.ndarray, bbox: List[int], 
                    target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """
        Извлечение и выравнивание области лица
        
        Args:
            image: Исходное изображение
            bbox: Координаты bounding box [x, y, width, height]
            target_size: Размер выходного изображения
        
        Returns:
            Выровненное изображение лица
        """
        x, y, w, h = bbox
        
        # Добавляем запас вокруг лица
        margin = 0.2
        x = max(0, int(x - margin * w))
        y = max(0, int(y - margin * h))
        w = min(image.shape[1] - x, int(w * (1 + 2 * margin)))
        h = min(image.shape[0] - y, int(h * (1 + 2 * margin)))
        
        # Извлекаем область лица
        face_region = image[y:y+h, x:x+w]
        
        if face_region.size == 0:
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        
        # Ресайз до target_size
        face_resized = cv2.resize(face_region, target_size)
        
        return face_resized
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Рисует bounding boxes на изображении
        
        Args:
            image: Исходное изображение
            detections: Результаты детекции
        
        Returns:
            Изображение с bounding boxes
        """
        result = image.copy()
        
        for det in detections:
            x, y, w, h = det['bbox']
            confidence = det.get('confidence', 0)
            
            # Рисуем bounding box
            color = (0, 255, 0)  # Зелёный
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            
            # Добавляем текст с confidence
            label = f"Face: {confidence:.2f}"
            cv2.putText(result, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return result