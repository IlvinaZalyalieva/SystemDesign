"""
Конфигурация системы распознавания лиц
"""

from typing import List, Dict, Any
import json
import os

class Settings:
    # Пороги для принятия решений (настроены для демо)
    # В демо используем более низкие пороги, так как mock данные не идеальны
    QUALITY_THRESHOLD_ALLOW: float = 0.65  # было 0.8
    QUALITY_THRESHOLD_REVIEW: float = 0.55  # было 0.7
    
    LIVENESS_THRESHOLD_ALLOW: float = 0.7  # было 0.9
    LIVENESS_THRESHOLD_REVIEW: float = 0.6  # было 0.8
    
    MATCH_THRESHOLD_ALLOW: float = 0.7  # было 0.85 - для демо с mock данными
    MATCH_THRESHOLD_REVIEW: float = 0.5  # было 0.75
    
    MARGIN_THRESHOLD_ALLOW: float = 0.05  # было 0.1
    MARGIN_THRESHOLD_REVIEW: float = 0.02  # было 0.05
    
    # Пути к данным
    DEMO_IMAGES_DIR: str = "demo/images"
    EMPLOYEE_DB_PATH: str = "data/employees.json"
    EMBEDDINGS_PATH: str = "data/embeddings.npy"
    EMBEDDINGS_INDEX_PATH: str = "data/embeddings_index.json"
    
    # Настройки API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Mock настройки
    USE_MOCK_DETECTION: bool = True  # Используем mock для детекции
    USE_MOCK_LIVENESS: bool = True
    USE_MOCK_EMBEDDINGS: bool = True

# Загружаем конфигурацию сотрудников
class EmployeeDB:
    def __init__(self, db_path: str = "data/employees.json"):
        self.db_path = db_path
        self.employees = self._load_employees()
    
    def _load_employees(self) -> Dict[str, Dict[str, Any]]:
        """Загрузка базы сотрудников из JSON файла"""
        import json
        import os
        
        if not os.path.exists(self.db_path):
            # Создаём демо-базу сотрудников
            return self._create_demo_db()
        
        with open(self.db_path, 'r') as f:
            return json.load(f)
    
    def _create_demo_db(self) -> Dict[str, Dict[str, Any]]:
        """Создание демо-базы сотрудников"""
        demo_db = {
            "emp-001": {
                "name": "Иван Петров",
                "department": "Разработка",
                "status": "active",
                "access_policy": {
                    "allowed_gates": ["gate-1", "gate-2", "gate-3"],
                    "working_hours": "08:00-20:00",
                    "restricted_zones": []
                }
            },
            "emp-002": {
                "name": "Мария Сидорова",
                "department": "Маркетинг",
                "status": "active",
                "access_policy": {
                    "allowed_gates": ["gate-1", "gate-2"],
                    "working_hours": "09:00-18:00",
                    "restricted_zones": ["lab-1"]
                }
            },
            "emp-003": {
                "name": "Алексей Иванов",
                "department": "Безопасность",
                "status": "inactive",  # Уволен
                "access_policy": {
                    "allowed_gates": [],
                    "working_hours": "00:00-00:00",
                    "restricted_zones": ["all"]
                }
            },
            "emp-004": {
                "name": "Елена Кузнецова",
                "department": "HR",
                "status": "active",
                "access_policy": {
                    "allowed_gates": ["gate-1", "gate-2", "gate-3"],
                    "working_hours": "08:30-17:30",
                    "restricted_zones": []
                }
            },
            "emp-005": {
                "name": "Дмитрий Смирнов",
                "department": "Финансы",
                "status": "active",
                "access_policy": {
                    "allowed_gates": ["gate-2"],
                    "working_hours": "10:00-19:00",
                    "restricted_zones": ["server-room"]
                }
            }
        }
        
        # Сохраняем демо-базу
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, 'w') as f:
            json.dump(demo_db, f, indent=2, ensure_ascii=False)
        
        return demo_db
    
    def get_employee(self, employee_id: str) -> Dict[str, Any]:
        """Получение информации о сотруднике"""
        return self.employees.get(employee_id, {})
    
    def is_active(self, employee_id: str) -> bool:
        """Проверка активности сотрудника"""
        employee = self.get_employee(employee_id)
        return employee.get("status") == "active"
    
    def can_access_gate(self, employee_id: str, gate_id: str) -> bool:
        """Проверка доступа сотрудника к проходной"""
        employee = self.get_employee(employee_id)
        if not employee:
            return False
        
        allowed_gates = employee.get("access_policy", {}).get("allowed_gates", [])
        return gate_id in allowed_gates

settings = Settings()
employee_db = EmployeeDB(settings.EMPLOYEE_DB_PATH)