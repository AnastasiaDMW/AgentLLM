# AgentLLM

Персональный ИИ-ассистент для управления заметками с использованием локальной модели Qwen.  
Проект позволяет:

- Создавать, редактировать и удалять заметки
- Автоматически сохранять заметки в `notes_store.json`
- Отвечать на вопросы, используя только сохранённые заметки
- Наблюдать за процессом генерации и запросами через Langfuse

Проект демонстрирует применение LLM для персональных задач с observability.

## **Запуск проекта**
### Клонировать репозиторий
```
git clone https://github.com/AnastasiaDMW/ML-Series-2.git
```
### Создать файл .env в корне и установить следующее переменные окружения
```
LANGFUSE_SECRET_KEY = {Ваш секретный токен}
LANGFUSE_PUBLIC_KEY = {Ваш публичный токен}
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
MODEL_NAME="Qwen/Qwen3-1.7B"
PERSIST_PATH="notes_store.json"
```
### Создать и активировать виртуальное окружение
```
# Windows
python -m venv venv
venv\Scripts\activate

# Linux, macOS
python3 -m venv venv
source venv/bin/activate
```
### Установить зависимости
```
pip install -r requirements.txt
```
### Запуск
Из корня проекта выполните:
```
python local_notes_chat_agent.py
```
Автор: <b>*AnastasiaDMW*</b>
