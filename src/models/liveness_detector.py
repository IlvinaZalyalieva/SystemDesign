"""
Модуль проверки liveness (анти-spoofing)
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import cv2
import random

class LivenessDetector:
    """Класс для проверки liveness (отличие реального лица от фото/видео)"""
    
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        
        if not use_mock:
            # В production здесь будет загрузка модели
            self.model = None
            # Пока используем mock
            self.use_mock = True
    
    def check_liveness(self, image: np.ndarray, 
                      face_detections: List[Dict[str, Any]],
                      previous_frames: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Проверка liveness для всех лиц на кадре
        
        Args:
            image: Текущий кадр
            face_detections: Результаты детекции лиц
            previous_frames: Предыдущие кадры для temporal анализа
        
        Returns:
            Словарь с результатами проверки liveness
        """
        if self.use_mock:
            return self._mock_liveness_check(image, face_detections, previous_frames)
        
        # В production здесь будет реальная проверка
        return self._mock_liveness_check(image, face_detections, previous_frames)
    
    def _mock_liveness_check(self, image: np.ndarray,
                           face_detections: List[Dict[str, Any]],
                           previous_frames: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Mock проверка liveness для демо
        
        В реальной системе здесь будут:
        1. Проверка моргания (eye aspect ratio changes)
        2. Анализ текстуры кожи
        3. Проверка отражений (спекулярных highlights)
        4. Глубина (если есть stereo/IR камера)
        5. Микродвижения головы
        """
        # Определяем, это happy path или risky path по качеству изображения
        height, width = image.shape[:2]
        avg_brightness = np.mean(image)
        
        # Для happy path (яркое изображение) даём высокий score
        if avg_brightness > 150 and len(face_detections) > 0:
            results = {
                'overall_score': 0.85,  # Высокий score для happy path
                'is_live': True,
                'spoof_type': None,
                'detection_reasons': ['good_brightness', 'natural_skin_texture'],
                'per_face_results': []
            }
        else:
            # Для risky path (тёмное/размытое изображение) даём низкий score
            results = {
                'overall_score': 0.45,  # Низкий score для risky path
                'is_live': False,
                'spoof_type': 'low_quality',
                'detection_reasons': ['low_brightness', 'possible_spoof'],
                'per_face_results': []
            }
        
        if not face_detections:
            results['overall_score'] = 0.0
            results['is_live'] = False
            results['detection_reasons'].append('no_faces_detected')
            return results
        
        # Для каждого лица
        for i, face in enumerate(face_detections):
            # Mock проверка разных типов spoofing
            
            # 1. Проверка яркости и контраста (фото может быть слишком ровным)
            face_region = self._extract_face_region(image, face['bbox'])
            if face_region is not None:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray_face)
                contrast = np.std(gray_face)
                
                # Фото на экране часто имеет специфичный контраст
                is_screen_spoof = (brightness > 180 and contrast < 25)
                is_print_spoof = (brightness < 80 and contrast < 20)
                
                if is_screen_spoof:
                    results['overall_score'] = 0.3
                    results['is_live'] = False
                    results['spoof_type'] = 'screen'
                    results['detection_reasons'].append('screen_spoof_detected')
                elif is_print_spoof:
                    results['overall_score'] = 0.4
                    results['is_live'] = False
                    results['spoof_type'] = 'print'
                    results['detection_reasons'].append('print_spoof_detected')
            
            # 2. Проверка размера и положения (маска может быть слишком большой)
            bbox = face['bbox']
            height, width = image.shape[:2]
            face_size_ratio = (bbox[2] * bbox[3]) / (height * width)
            
            if face_size_ratio > 0.4:  # Слишком большое лицо
                results['overall_score'] = min(results['overall_score'], 0.6)
                results['detection_reasons'].append('unusual_face_size')
            
            # 3. Проверка landmarks (у реального лица должны быть нормальные пропорции)
            landmarks = face.get('landmarks', {})
            if landmarks:
                has_valid_proportions = self._check_face_proportions(landmarks)
                if not has_valid_proportions:
                    results['overall_score'] = min(results['overall_score'], 0.5)
                    results['detection_reasons'].append('abnormal_face_proportions')
            
            # Результат для конкретного лица
            face_result = {
                'face_index': i,
                'liveness_score': results['overall_score'],
                'is_live': results['is_live'],
                'reasons': results['detection_reasons'].copy()
            }
            results['per_face_results'].append(face_result)
        
        # Если не найдено spoofing, добавляем положительные причины
        if results['is_live']:
            results['detection_reasons'].extend([
                'normal_brightness_pattern',
                'natural_face_proportions',
                'expected_texture_variation'
            ])
        
        # Добавляем случайный шум для реалистичности
        if results['overall_score'] > 0.8:
            # 5% случаев ложный negative (принятие spoof за реальное)
            if random.random() < 0.05:
                results['overall_score'] = random.uniform(0.9, 0.95)
                results['is_live'] = True
            else:
                results['overall_score'] = random.uniform(0.85, 0.98)
        elif results['overall_score'] < 0.6:
            # 2% случаев ложный positive (отказ реальному лицу)
            if random.random() < 0.02:
                results['overall_score'] = random.uniform(0.7, 0.8)
                results['is_live'] = False
                results['detection_reasons'].append('false_negative_in_mock')
        
        return results
    
    def _extract_face_region(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Извлечение области лица"""
        x, y, w, h = bbox
        height, width = image.shape[:2]
        
        # Проверка границ
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return image[y1:y2, x1:x2]
    
    def _check_face_proportions(self, landmarks: Dict[str, Tuple[float, float]]) -> bool:
        """Проверка пропорций лица по landmarks"""
        required_points = ['left_eye', 'right_eye', 'nose']
        
        # Проверяем наличие всех необходимых точек
        if not all(point in landmarks for point in required_points):
            return True  # Не можем проверить
        
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        nose = landmarks['nose']
        
        # Расстояние между глазами
        eye_distance = ((right_eye[0] - left_eye[0]) ** 2 + 
                       (right_eye[1] - left_eye[1]) ** 2) ** 0.5
        
        # Расстояние от центра между глазами до носа
        eye_center = ((left_eye[0] + right_eye[0]) / 2,
                     (left_eye[1] + right_eye[1]) / 2)
        eye_nose_distance = ((nose[0] - eye_center[0]) ** 2 + 
                           (nose[1] - eye_center[1]) ** 2) ** 0.5
        
        # У реального лица соотношение eye_distance / eye_nose_distance примерно 1:1
        ratio = eye_distance / (eye_nose_distance + 1e-6)
        
        # Нормальное соотношение: 0.8 - 1.2
        return 0.7 <= ratio <= 1.3
    
    def get_liveness_reasons(self, liveness_result: Dict[str, Any]) -> List[str]:
        """Генерация причин для результата liveness проверки"""
        reasons = liveness_result.get('detection_reasons', [])
        
        score = liveness_result.get('overall_score', 0)
        is_live = liveness_result.get('is_live', False)
        
        if is_live and score >= 0.9:
            if 'liveness_ok' not in reasons:
                reasons.append('liveness_ok')
        elif not is_live:
            if 'liveness_failed' not in reasons:
                reasons.append('liveness_failed')
            
            spoof_type = liveness_result.get('spoof_type')
            if spoof_type:
                reasons.append(f'spoof_type_{spoof_type}')
        
        return reasons