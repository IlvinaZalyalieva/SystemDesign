"""
Модуль оценки качества кадра и лица
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
import math

class QualityAssessor:
    """Класс для оценки качества кадра и лица"""
    
    def __init__(self):
        # Пороги для разных аспектов качества
        self.thresholds = {
            'brightness': {'min': 50, 'max': 200, 'ideal': 120},
            'contrast': {'min': 30, 'ideal': 100},
            'sharpness': {'min': 10, 'ideal': 50},
            'face_size': {'min': 0.1, 'ideal': 0.3},  # доля от высоты кадра
            'face_yaw': {'max': 30},  # градусы
            'face_pitch': {'max': 20},  # градусы
            'occlusion': {'max': 0.2}  # доля закрытого лица
        }
    
    def assess_frame_quality(self, image: np.ndarray, 
                           face_detections: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Оценка общего качества кадра
        
        Args:
            image: Изображение кадра
            face_detections: Результаты детекции лиц (опционально)
        
        Returns:
            Словарь с оценками качества
        """
        height, width = image.shape[:2]
        
        # Оценка яркости
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_score = self._score_brightness(brightness)
        
        # Оценка контраста
        contrast = np.std(gray)
        contrast_score = self._score_contrast(contrast)
        
        # Оценка резкости (лапласиан)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = self._score_sharpness(sharpness)
        
        # Оценка наличия лиц
        face_score = 0.0
        face_metrics = {}
        
        if face_detections:
            if len(face_detections) == 0:
                face_score = 0.0
                face_metrics['no_faces'] = True
            else:
                # Берём первое (самое большое/уверенное) лицо
                main_face = face_detections[0]
                face_score, face_metrics = self._assess_face_quality(image, main_face)
        
        # Общий score как взвешенная сумма
        weights = {
            'brightness': 0.2,
            'contrast': 0.2,
            'sharpness': 0.3,
            'face': 0.3
        }
        
        overall_score = (
            brightness_score * weights['brightness'] +
            contrast_score * weights['contrast'] +
            sharpness_score * weights['sharpness'] +
            face_score * weights['face']
        )
        
        # Классификация качества
        if overall_score >= 0.8:
            quality_class = "good"
        elif overall_score >= 0.6:
            quality_class = "acceptable"
        else:
            quality_class = "poor"
        
        return {
            'overall_score': float(overall_score),
            'quality_class': quality_class,
            'component_scores': {
                'brightness': float(brightness_score),
                'contrast': float(contrast_score),
                'sharpness': float(sharpness_score),
                'face': float(face_score)
            },
            'face_metrics': face_metrics,
            'raw_metrics': {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'sharpness': float(sharpness)
            }
        }
    
    def _assess_face_quality(self, image: np.ndarray, face_detection: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Оценка качества конкретного лица"""
        height, width = image.shape[:2]
        bbox = face_detection['bbox']
        landmarks = face_detection.get('landmarks', {})
        
        x, y, w, h = bbox
        
        # Размер лица относительно кадра
        face_height_ratio = h / height
        size_score = self._score_face_size(face_height_ratio)
        
        # Оценка позы (yaw, pitch, roll) по landmarks
        pose_scores = self._estimate_head_pose(landmarks, image.shape)
        pose_score = np.mean(list(pose_scores.values())) if pose_scores else 0.7
        
        # Оценка occlusion (упрощённо)
        occlusion_score = 0.9  # Mock - в реальности нужна модель
        
        # Комбинированный score
        face_score = size_score * 0.4 + pose_score * 0.4 + occlusion_score * 0.2
        
        face_metrics = {
            'size_ratio': float(face_height_ratio),
            'size_score': float(size_score),
            'pose_scores': pose_scores,
            'pose_score': float(pose_score),
            'occlusion_score': float(occlusion_score)
        }
        
        return float(face_score), face_metrics
    
    def _score_brightness(self, brightness: float) -> float:
        """Оценка яркости"""
        ideal = self.thresholds['brightness']['ideal']
        min_val = self.thresholds['brightness']['min']
        max_val = self.thresholds['brightness']['max']
        
        if min_val <= brightness <= max_val:
            # Гауссова функция вокруг идеального значения
            sigma = (max_val - min_val) / 6
            score = math.exp(-0.5 * ((brightness - ideal) / sigma) ** 2)
            return max(0.1, min(1.0, score))
        else:
            return 0.1
    
    def _score_contrast(self, contrast: float) -> float:
        """Оценка контраста"""
        ideal = self.thresholds['contrast']['ideal']
        min_val = self.thresholds['contrast']['min']
        
        if contrast >= min_val:
            # Логарифмическая шкала
            score = min(1.0, contrast / ideal)
            return max(0.1, score)
        else:
            return 0.1
    
    def _score_sharpness(self, sharpness: float) -> float:
        """Оценка резкости"""
        ideal = self.thresholds['sharpness']['ideal']
        min_val = self.thresholds['sharpness']['min']
        
        if sharpness >= min_val:
            score = min(1.0, sharpness / ideal)
            return max(0.1, score)
        else:
            return 0.1
    
    def _score_face_size(self, size_ratio: float) -> float:
        """Оценка размера лица"""
        ideal = self.thresholds['face_size']['ideal']
        min_val = self.thresholds['face_size']['min']
        
        if size_ratio >= min_val:
            # Предпочтительнее лица побольше, но не слишком большие
            if size_ratio <= ideal:
                score = size_ratio / ideal
            else:
                score = max(0.1, 1.0 - (size_ratio - ideal) / (1.0 - ideal))
            return max(0.1, min(1.0, score))
        else:
            return 0.1
    
    def _estimate_head_pose(self, landmarks: Dict[str, Tuple[float, float]], 
                          image_shape: Tuple[int, int]) -> Dict[str, float]:
        """
        Оценка позы головы по landmarks
        Упрощённая версия для демо
        """
        if not landmarks:
            return {'yaw': 0.8, 'pitch': 0.8, 'roll': 0.9}
        
        scores = {}
        
        # Mock оценка - в реальности нужен solvePnP
        if 'left_eye' in landmarks and 'right_eye' in landmarks:
            # Оценка yaw по положению глаз
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            
            eye_distance = abs(right_eye[0] - left_eye[0])
            face_center_x = (left_eye[0] + right_eye[0]) / 2
            image_center_x = image_shape[1] / 2
            
            # Отклонение от центра
            x_offset = abs(face_center_x - image_center_x) / (image_shape[1] / 2)
            yaw_score = max(0.1, 1.0 - x_offset)
            scores['yaw'] = yaw_score
            
            # Оценка pitch по положению носа относительно глаз
            if 'nose' in landmarks:
                nose = landmarks['nose']
                eyes_y = (left_eye[1] + right_eye[1]) / 2
                vertical_offset = abs(nose[1] - eyes_y) / (image_shape[0] / 4)
                pitch_score = max(0.1, 1.0 - vertical_offset)
                scores['pitch'] = pitch_score
            
            # Оценка roll по углу между глазами
            if left_eye[0] != right_eye[0]:
                angle = math.degrees(math.atan2(
                    right_eye[1] - left_eye[1],
                    right_eye[0] - left_eye[0]
                ))
                roll_deviation = min(abs(angle), abs(angle - 180))
                roll_score = max(0.1, 1.0 - roll_deviation / 45)  # 45° максимум
                scores['roll'] = roll_score
        
        return scores
    
    def get_quality_reasons(self, quality_assessment: Dict[str, Any]) -> List[str]:
        """Генерация причин оценки качества"""
        reasons = []
        
        overall_score = quality_assessment['overall_score']
        component_scores = quality_assessment['component_scores']
        quality_class = quality_assessment['quality_class']
        
        reasons.append(f"overall_quality_{quality_class}")
        
        # Проверяем компоненты
        if component_scores['brightness'] < 0.6:
            reasons.append("low_brightness")
        
        if component_scores['contrast'] < 0.6:
            reasons.append("low_contrast")
        
        if component_scores['sharpness'] < 0.6:
            reasons.append("low_sharpness")
        
        if component_scores['face'] < 0.6:
            reasons.append("poor_face_quality")
        
        # Проверяем лицо
        face_metrics = quality_assessment.get('face_metrics', {})
        if face_metrics.get('size_score', 1.0) < 0.5:
            reasons.append("face_too_small")
        
        if face_metrics.get('pose_score', 1.0) < 0.6:
            reasons.append("bad_head_pose")
        
        # Если всё хорошо
        if overall_score >= 0.8 and len(reasons) == 1:
            reasons.append("quality_ok")
        
        return reasons