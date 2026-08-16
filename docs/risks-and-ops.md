# Риски и операционные аспекты

## Low-latency, надёжность и деградация

### Edge-вычисления vs центральный сервис
- **Edge:** детекция лица, quality assessment, liveness, эмбеддинг, локальный поиск
- **Центр:** полный поиск 1:N, access policy, audit log, управление моделями
- **Почему:** Edge для latency (<200 мс), центр для consistency и scale

### Достижение 1 секунды p95
- **Pipeline оптимизация:** параллельные этапы где возможно
- **Model quantization:** FP16/INT8 для inference
- **Cache warming:** предзагрузка эмбеддингов активных сотрудников
- **Connection pooling:** keep-alive к центральному сервису

### Кэширование базы эмбеддингов на проходной
- **Размер:** топ-10k активных сотрудников (80% трафика)
- **Обновление:** incremental каждые 5 минут при наличии сети
- **TTL:** 24 часа для неактивных записей
- **Версионность:** A/B тестирование новых версий кэша

### Поведение при потере сети
1. **Degraded mode:** только локальный кэш
2. **Новые/уволенные сотрудники → manual_review**
3. **Локальное логирование с репликацией при восстановлении**
4. **Health checks каждые 30 секунд**

### Поведение при недоступности модели/базы
- **Fallback модели:** lightweight версии для критичных компонентов
- **Circuit breaker:** отказ компонента → manual_review для всего трафика
- **Graceful degradation:** снижение quality thresholds при partial failures

### Идемпотентность команд турникету
- **Unique event_id** для каждого решения
- **Deduplication на стороне турникета** (5-секундное окно)
- **Подтверждение выполнения команды**
- **Retry с exponential backoff при таймаутах**

### Защита от двойного открытия
- **Session-based:** один проход = одна сессия (вход+выход отдельно)
- **Временное окно:** минимум 30 секунд между проходами одного сотрудника
- **Физические датчики:** infrared beams для детекции tailgating
- **Логирование всех попыток** с timestamp и решением

## Privacy, safety и governance

### Биометрические данные
- **Что считается биометрикой:** эмбеддинги лица (512-dim векторы)
- **Исходные изображения:** хранятся 30 дней в шифрованном виде для audit
- **Эмбеддинги:** хранятся бессрочно (пока сотрудник активен)
- **Шифрование:** at-rest AES-256, in-transit TLS 1.3

### Доступ к базе шаблонов
- **Role-based access control:**
  - ML team: read-only для retraining
  - Security: read-only для расследований
  - HR: add/remove при найме/увольнении
  - Ops: мониторинг, no data access
- **Audit log всех операций:** кто, когда, что сделал
- **Multi-factor authentication для sensitive операций**

### Audit log
- **Что логируется:** все события доступа с полным контекстом
- **Retention:** 7 лет для compliance
- **Immutability:** write-once хранилище с cryptographic hashing
- **Search:** полнотекстовый поиск по всем полям
- **Export:** API для compliance проверок

### Удаление сотрудника и отзыв согласия
1. **Immediate revocation:** флаг в центральной базе
2. **Propagation to edge:** в течение 5 минут через push
3. **Emergency revocation:** принудительный push за 30 секунд
4. **Offline handling:** уволенные в degraded mode → manual_review
5. **Data deletion:** через 30 дней после увольнения (кроме audit log)

### Защита от model inversion
- **Эмбеддинги вместо исходных изображений** для поиска
- **Differential privacy** при обучении моделей
- **Rate limiting** запросов к API поиска
- **Monitoring необычных паттернов** запросов

### Решения, требующие человеческого контроля
1. **Borderline cases:** match_score в диапазоне manual_review
2. **Low quality:** quality_score < 0.7
3. **Suspicious liveness:** liveness_score < 0.8
4. **Offline mode:** все события при потере сети
5. **Multiple attempts:** >3 попыток за 5 минут

### Объяснение причин отказа
- **Employee-facing:** общие категории (плохое качество, не распознан)
- **Security-facing:** детальные причины для расследований
- **No sensitive info:** не показывать similarity scores или других кандидатов
- **Self-service portal:** история своих проходов с анонимизированными причинами

### Правовые и организационные аспекты
- **Явное согласие сотрудников** на обработку биометрии
- **Data Protection Officer** для compliance
- **Regular security audits** (раз в квартал)
- **Incident response plan** для data breaches
- **Employee training** по privacy и security