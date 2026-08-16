"""
FastAPI сервер для демонстрации API системы распознавания лиц
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import cv2
import numpy as np
import json
import os
import sys
from datetime import datetime
import uuid

# Добавляем путь к src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.face_detector import FaceDetector
from src.models.quality_assessor import QualityAssessor
from src.models.liveness_detector import LivenessDetector
from src.services.face_matcher import FaceMatcher
from src.services.decision_maker import DecisionMaker
from src.config import settings, employee_db

# Модели данных
class AccessEvent(BaseModel):
    event_id: str
    gate_id: str
    camera_id: str
    captured_at: str
    frame_uri: Optional[str] = None
    metadata: Dict[str, Any] = {}

class AccessResponse(BaseModel):
    event_id: str
    decision_id: str
    decision: str  # allow, deny, manual_review
    employee_id: Optional[str] = None
    match_score: float
    margin_to_second_best: float
    quality: Dict[str, Any]
    reasons: list[str]
    turnstile_command: str
    requires_human_review: bool
    degraded_mode: bool
    audit_id: str
    latency_ms: int

# Инициализация приложения
app = FastAPI(
    title="Face Recognition Access Control API",
    description="API для системы распознавания лиц на проходной",
    version="1.0.0"
)

# Инициализация компонентов
face_detector = FaceDetector(use_mock=settings.USE_MOCK_DETECTION)
quality_assessor = QualityAssessor()
liveness_detector = LivenessDetector(use_mock=settings.USE_MOCK_LIVENESS)
face_matcher = FaceMatcher()
decision_maker = DecisionMaker()

@app.get("/")
async def root():
    """Корневой endpoint с информацией о API"""
    return {
        "service": "Face Recognition Access Control",
        "version": "1.0.0",
        "endpoints": {
            "POST /v1/access/verify": "Проверка доступа по изображению",
            "GET /v1/health": "Проверка здоровья системы",
            "GET /v1/employees": "Список сотрудников (демо)"
        }
    }

@app.get("/v1/health")
async def health_check():
    """Проверка здоровья системы"""
    components = {
        "face_detector": "ok",
        "quality_assessor": "ok", 
        "liveness_detector": "ok",
        "face_matcher": "ok",
        "decision_maker": "ok"
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": components
    }

@app.get("/v1/employees")
async def list_employees():
    """Список сотрудников в демо-базе"""
    employees = employee_db.employees
    return {
        "count": len(employees),
        "employees": employees
    }

@app.post("/v1/access/verify", response_model=AccessResponse)
async def verify_access(
    event: AccessEvent,
    image_file: UploadFile = File(None)
):
    """
    Проверка доступа по изображению с камеры
    
    Можно передать либо frame_uri, либо image_file
    """
    try:
        # Засекаем время начала обработки
        import time
        start_time = time.time()
        
        # Получаем изображение
        image = await _get_image(event, image_file)
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Не удалось получить изображение. Укажите frame_uri или загрузите image_file"
            )
        
        # Конвертируем event в dict
        event_data = event.dict()
        
        # Обрабатываем событие
        print(f"Обработка события {event_data.get('event_id')}...")
        
        # Шаг 1: Детекция лиц
        face_detections = face_detector.detect_faces(image)
        
        # Шаг 2: Оценка качества
        quality_result = quality_assessor.assess_frame_quality(image, face_detections)
        quality_result['reasons'] = quality_assessor.get_quality_reasons(quality_result)
        
        # Шаг 3: Проверка liveness
        liveness_result = liveness_detector.check_liveness(image, face_detections)
        liveness_result['detection_reasons'] = liveness_detector.get_liveness_reasons(liveness_result)
        
        # Шаг 4: Matching
        match_result = None
        if face_detections:
            # Mock эмбеддинг для демо
            mock_embedding = np.random.randn(512)
            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
            
            match_result = face_matcher.match_face(mock_embedding)
            match_result['reasons'] = face_matcher.get_match_reasons(
                match_result, decision_maker.thresholds
            )
        else:
            match_result = {
                'best_match': None,
                'best_score': 0.0,
                'margin': 0.0,
                'reasons': ['no_faces_detected']
            }
        
        # Шаг 5: Принятие решения
        decision = decision_maker.make_decision(
            event_data, quality_result, liveness_result, match_result
        )
        
        # Шаг 6: Логирование
        decision_maker.log_decision(decision)
        
        # Вычисляем latency
        end_time = time.time()
        decision['latency_ms'] = int((end_time - start_time) * 1000)
        
        # Преобразуем в response модель
        response = AccessResponse(**decision)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.post("/v1/demo/happy-path")
async def demo_happy_path():
    """Демонстрация happy path сценария"""
    try:
        # Создаём mock событие
        event_data = {
            "event_id": f"demo-happy-{str(uuid.uuid4())[:8]}",
            "gate_id": "gate-2",
            "camera_id": "cam-2a",
            "captured_at": datetime.now().isoformat() + "Z",
            "frame_uri": "demo://happy-path",
            "metadata": {
                "direction": "in",
                "illumination": "normal",
                "edge_node": "edge-gate-2",
                "network": "online"
            }
        }
        
        # Создаём mock изображение хорошего качества
        image = _create_demo_image(good_quality=True)
        
        # Обрабатываем
        event = AccessEvent(**event_data)
        
        # Имитируем вызов verify_access
        import time
        start_time = time.time()
        
        face_detections = face_detector.detect_faces(image)
        quality_result = quality_assessor.assess_frame_quality(image, face_detections)
        quality_result['reasons'] = quality_assessor.get_quality_reasons(quality_result)
        
        liveness_result = liveness_detector.check_liveness(image, face_detections)
        liveness_result['detection_reasons'] = liveness_detector.get_liveness_reasons(liveness_result)
        
        if face_detections:
            mock_embedding = np.random.randn(512)
            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
            match_result = face_matcher.match_face(mock_embedding)
            match_result['reasons'] = face_matcher.get_match_reasons(
                match_result, decision_maker.thresholds
            )
        else:
            match_result = {
                'best_match': None,
                'best_score': 0.0,
                'margin': 0.0,
                'reasons': ['no_faces_detected']
            }
        
        decision = decision_maker.make_decision(
            event_data, quality_result, liveness_result, match_result
        )
        
        decision_maker.log_decision(decision)
        
        end_time = time.time()
        decision['latency_ms'] = int((end_time - start_time) * 1000)
        
        response = AccessResponse(**decision)
        
        return {
            "scenario": "happy_path",
            "description": "Успешное распознавание сотрудника в хороших условиях",
            "result": response.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка демо: {str(e)}")

@app.post("/v1/demo/risky-path")
async def demo_risky_path():
    """Демонстрация risky path сценария"""
    try:
        # Создаём mock событие с плохим качеством
        event_data = {
            "event_id": f"demo-risky-{str(uuid.uuid4())[:8]}",
            "gate_id": "gate-1",
            "camera_id": "cam-1b",
            "captured_at": datetime.now().isoformat() + "Z",
            "frame_uri": "demo://risky-path",
            "metadata": {
                "direction": "in",
                "illumination": "backlight",
                "occlusion_hint": "mask",
                "edge_node": "edge-gate-1",
                "network": "online"
            }
        }
        
        # Создаём mock изображение плохого качества
        image = _create_demo_image(good_quality=False)
        
        # Обрабатываем
        import time
        start_time = time.time()
        
        face_detections = face_detector.detect_faces(image)
        quality_result = quality_assessor.assess_frame_quality(image, face_detections)
        quality_result['reasons'] = quality_assessor.get_quality_reasons(quality_result)
        
        liveness_result = liveness_detector.check_liveness(image, face_detections)
        liveness_result['detection_reasons'] = liveness_detector.get_liveness_reasons(liveness_result)
        
        if face_detections:
            mock_embedding = np.random.randn(512)
            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
            match_result = face_matcher.match_face(mock_embedding)
            match_result['reasons'] = face_matcher.get_match_reasons(
                match_result, decision_maker.thresholds
            )
        else:
            match_result = {
                'best_match': None,
                'best_score': 0.0,
                'margin': 0.0,
                'reasons': ['no_faces_detected']
            }
        
        decision = decision_maker.make_decision(
            event_data, quality_result, liveness_result, match_result
        )
        
        decision_maker.log_decision(decision)
        
        end_time = time.time()
        decision['latency_ms'] = int((end_time - start_time) * 1000)
        
        response = AccessResponse(**decision)
        
        return {
            "scenario": "risky_path",
            "description": "Сомнительный случай с низким качеством кадра → ручная проверка",
            "result": response.dict(),
            "human_review_required": decision.get('decision') == 'manual_review'
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка демо: {str(e)}")

async def _get_image(event: AccessEvent, image_file: UploadFile) -> Optional[np.ndarray]:
    """Получение изображения из разных источников"""
    if image_file:
        # Читаем загруженный файл
        contents = await image_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    elif event.frame_uri:
        # Пытаемся загрузить по URI
        if event.frame_uri.startswith("file://"):
            file_path = event.frame_uri[7:]  # Убираем "file://"
            if os.path.exists(file_path):
                return cv2.imread(file_path)
        elif event.frame_uri.startswith("demo://"):
            # Демо изображение
            if "happy" in event.frame_uri:
                return _create_demo_image(good_quality=True)
            else:
                return _create_demo_image(good_quality=False)
    
    return None

def _create_demo_image(good_quality: bool = True) -> np.ndarray:
    """Создание демо изображения"""
    height, width = 480, 640
    
    if good_quality:
        image = np.random.randint(100, 200, (height, width, 3), dtype=np.uint8)
        
        # Добавляем "лицо"
        face_center = (width // 2, height // 2)
        face_radius = min(width, height) // 6
        
        cv2.ellipse(image, face_center, (face_radius, int(face_radius * 1.2)), 
                   0, 0, 360, (200, 150, 100), -1)
        
        eye_y = face_center[1] - face_radius // 3
        left_eye = (face_center[0] - face_radius // 2, eye_y)
        right_eye = (face_center[0] + face_radius // 2, eye_y)
        cv2.circle(image, left_eye, face_radius // 6, (50, 50, 50), -1)
        cv2.circle(image, right_eye, face_radius // 6, (50, 50, 50), -1)
        
        mouth_y = face_center[1] + face_radius // 3
        cv2.ellipse(image, (face_center[0], mouth_y), 
                   (face_radius // 2, face_radius // 4), 0, 0, 180, (50, 50, 50), 2)
    else:
        image = np.random.randint(30, 80, (height, width, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (15, 15), 5)
        
        noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )