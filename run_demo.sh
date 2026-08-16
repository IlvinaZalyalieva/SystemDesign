#!/bin/bash

# Скрипт запуска демонстрации системы распознавания лиц

echo "Запуск демонстрации системы распознавания лиц на проходной"
echo "=========================================================="

# Проверяем наличие Python
if ! command -v python3 &> /dev/null; then
    echo "Ошибка: Python3 не найден"
    exit 1
fi

# Проверяем наличие виртуального окружения
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python3 -m venv venv
fi

# Активируем виртуальное окружение
echo "Активация виртуального окружения..."
source venv/bin/activate

# Устанавливаем зависимости
echo "Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

# Создаём необходимые директории
echo "Создание структуры директорий..."
mkdir -p logs data demo/images

echo ""
echo "Запуск демонстрационных сценариев..."
echo "===================================="

# Запускаем демо
python src/demo.py

echo ""
echo "Демонстрация завершена!"
echo "Логи сохранены в logs/access_log.json"
echo "Для запуска API сервера выполните: python src/api/main.py"
echo ""