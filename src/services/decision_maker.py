"""
Сервис принятия решений о доступе
"""

from typing import Dict, Any, List, Tuple
from datetime import datetime
import uuid
from src.config import settings, employee_db

class DecisionMaker:
    """Класс для принятия решений о доступе на основе всех проверок"""
    
    def __init__(self):
        self.thresholds = {
            'quality_allow': settings.QUALITY_THRESHOLD_ALLOW,
            'quality_review': settings.QUALITY_THRESHOLD_REVIEW,
            'liveness_allow': settings.LIVENESS_THRESHOLD_ALLOW,
            'liveness_review': settings.LIVENESS_THRESHOLD_REVIEW,
            'match_allow': settings.MATCH_THRESHOLD_ALLOW,
            'match_review': settings.MATCH_THRESHOLD_REVIEW,
            'margin_allow': settings.MARGIN_THRESHOLD_ALLOW,
            'margin_review': settings.MARGIN_THRESHOLD_REVIEW
        }
    
    def make_decision(self, event_data: Dict[str, Any],
                     quality_result: Dict[str, Any],
                     liveness_result: Dict[str, Any],
                     match_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Принятие итогового решения о доступе
        
        Args:
            event_data: Данные события с камеры
            quality_result: Результаты оценки качества
            liveness_result: Результаты проверки liveness
            match_result: Результаты matching с базой
        
        Returns:
            Итоговое решение с деталями
        """
        # Извлекаем ключевые метрики
        quality_score = quality_result.get('overall_score', 0.0)
        liveness_score = liveness_result.get('overall_score', 0.0)
        is_live = liveness_result.get('is_live', False)
        
        best_match = match_result.get('best_match')
        best_score = match_result.get('best_score', 0.0)
        margin = match_result.get('margin', 0.0)
        
        employee_id = best_match.get('employee_id') if best_match else None
        
        # Собираем все причины
        all_reasons = []
        
        # Причины из quality проверки
        quality_reasons = quality_result.get('reasons', [])
        all_reasons.extend(quality_reasons)
        
        # Причины из liveness проверки
        liveness_reasons = liveness_result.get('detection_reasons', [])
        all_reasons.extend(liveness_reasons)
        
        # Причины из matching
        match_reasons = match_result.get('reasons', [])
        all_reasons.extend(match_reasons)
        
        # Проверяем каждое условие
        conditions = []
        
        # 1. Качество кадра
        if quality_score >= self.thresholds['quality_allow']:
            conditions.append(('quality', 'pass', quality_score))
        elif quality_score >= self.thresholds['quality_review']:
            conditions.append(('quality', 'review', quality_score))
        else:
            conditions.append(('quality', 'fail', quality_score))
        
        # 2. Liveness
        if is_live and liveness_score >= self.thresholds['liveness_allow']:
            conditions.append(('liveness', 'pass', liveness_score))
        elif is_live and liveness_score >= self.thresholds['liveness_review']:
            conditions.append(('liveness', 'review', liveness_score))
        else:
            conditions.append(('liveness', 'fail', liveness_score))
        
        # 3. Matching
        if best_score >= self.thresholds['match_allow'] and margin >= self.thresholds['margin_allow']:
            conditions.append(('match', 'pass', best_score))
        elif best_score >= self.thresholds['match_review'] and margin >= self.thresholds['margin_review']:
            conditions.append(('match', 'review', best_score))
        else:
            conditions.append(('match', 'fail', best_score))
        
        # 4. Проверка доступа сотрудника
        access_check = self._check_employee_access(employee_id, event_data.get('gate_id'))
        conditions.append(('access', access_check['status'], access_check.get('score', 1.0)))
        if access_check.get('reasons'):
            all_reasons.extend(access_check['reasons'])
        
        # Определяем итоговое решение
        decision, requires_human_review, decision_reasons = self._determine_final_decision(
            conditions, all_reasons
        )
        
        # Генерируем уникальные ID
        decision_id = f"d-{str(uuid.uuid4())[:8]}"
        audit_id = f"a-{str(uuid.uuid4())[:8]}"
        
        # Mock latency для демо
        latency_ms = self._mock_latency(decision)
        
        # Формируем итоговый ответ
        result = {
            'event_id': event_data.get('event_id', 'unknown'),
            'decision_id': decision_id,
            'decision': decision,
            'employee_id': employee_id,
            'match_score': float(best_score),
            'margin_to_second_best': float(margin),
            'quality': {
                'face_detected': len(quality_result.get('face_detections', [])) > 0,
                'quality_score': float(quality_score),
                'liveness_score': float(liveness_score),
                'is_live': bool(is_live)
            },
            'reasons': decision_reasons,
            'turnstile_command': 'open' if decision == 'allow' else 'close',
            'requires_human_review': requires_human_review,
            'degraded_mode': event_data.get('metadata', {}).get('network') == 'offline',
            'audit_id': audit_id,
            'latency_ms': latency_ms,
            'conditions': conditions,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def _check_employee_access(self, employee_id: str, gate_id: str) -> Dict[str, Any]:
        """Проверка прав доступа сотрудника"""
        if not employee_id:
            return {'status': 'fail', 'score': 0.0, 'reasons': ['no_employee_matched']}
        
        # Проверяем активность сотрудника
        if not employee_db.is_active(employee_id):
            return {
                'status': 'fail',
                'score': 0.0,
                'reasons': ['employee_inactive', 'access_revoked']
            }
        
        # Проверяем доступ к конкретной проходной
        if gate_id and not employee_db.can_access_gate(employee_id, gate_id):
            return {
                'status': 'fail',
                'score': 0.0,
                'reasons': ['gate_not_allowed', 'access_restricted']
            }
        
        # Проверяем время работы (упрощённо)
        # В реальной системе здесь будет проверка графика работы
        
        return {'status': 'pass', 'score': 1.0, 'reasons': ['access_granted']}
    
    def _determine_final_decision(self, conditions: List[Tuple[str, str, float]],
                                all_reasons: List[str]) -> Tuple[str, bool, List[str]]:
        """
        Определение итогового решения на основе всех условий
        
        Логика:
        - Если любой компонент FAIL → DENY
        - Если 2+ компонентов REVIEW → MANUAL_REVIEW
        - Если 1 компонент REVIEW → зависит от важности компонента
        - Если все PASS → ALLOW
        """
        # Подсчитываем статусы
        status_counts = {'pass': 0, 'review': 0, 'fail': 0}
        for component, status, score in conditions:
            status_counts[status] += 1
        
        decision_reasons = []
        
        # Правило 1: Любой FAIL → DENY
        if status_counts['fail'] > 0:
            # Определяем, что именно упало
            failed_components = [c[0] for c in conditions if c[1] == 'fail']
            for comp in failed_components:
                decision_reasons.append(f'{comp}_failed')
            
            return 'deny', False, decision_reasons
        
        # Правило 2: 2+ REVIEW → MANUAL_REVIEW
        if status_counts['review'] >= 2:
            review_components = [c[0] for c in conditions if c[1] == 'review']
            for comp in review_components:
                decision_reasons.append(f'{comp}_needs_review')
            decision_reasons.append('multiple_review_conditions')
            
            return 'manual_review', True, decision_reasons
        
        # Правило 3: 1 REVIEW → зависит от компонента
        if status_counts['review'] == 1:
            review_component = [c[0] for c in conditions if c[1] == 'review'][0]
            
            # Критические компоненты всегда требуют review
            critical_components = ['liveness', 'access']
            if review_component in critical_components:
                decision_reasons.append(f'{review_component}_critical_review')
                return 'manual_review', True, decision_reasons
            
            # Для quality и match можно разрешить с high confidence
            review_score = [c[2] for c in conditions if c[0] == review_component and c[1] == 'review'][0]
            
            if review_component == 'match' and review_score > 0.8:
                decision_reasons.append(f'{review_component}_high_confidence')
                return 'allow', False, decision_reasons
            elif review_component == 'quality' and review_score > 0.75:
                decision_reasons.append(f'{review_component}_acceptable')
                return 'allow', False, decision_reasons
            else:
                decision_reasons.append(f'{review_component}_needs_review')
                return 'manual_review', True, decision_reasons
        
        # Правило 4: Все PASS → ALLOW
        if status_counts['pass'] == len(conditions):
            decision_reasons.append('all_checks_passed')
            return 'allow', False, decision_reasons
        
        # Fallback: MANUAL_REVIEW
        decision_reasons.append('uncertain_decision')
        return 'manual_review', True, decision_reasons
    
    def _mock_latency(self, decision: str) -> int:
        """Mock задержка для демо"""
        import random
        
        # Разная задержка для разных решений
        base_latency = {
            'allow': 400,
            'deny': 300,
            'manual_review': 500
        }
        
        base = base_latency.get(decision, 400)
        # Добавляем случайность
        latency = base + random.randint(-50, 150)
        
        return max(100, min(latency, 1000))  # Ограничиваем 100-1000 мс
    
    def log_decision(self, decision: Dict[str, Any], log_path: str = "logs/access_log.json"):
        """Логирование решения"""
        import json
        import os
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            **decision
        }
        
        # Записываем в файл
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error logging decision: {e}")
            
            # Fallback: запись в отдельный файл
            fallback_path = f"logs/access_log_fallback_{datetime.now().strftime('%Y%m%d')}.json"
            os.makedirs(os.path.dirname(fallback_path), exist_ok=True)
            with open(fallback_path, 'a') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')