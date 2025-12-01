import json
import os
import traceback
import datetime

from typing import Dict, Optional
from langfuse import get_client, observe
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME")
PERSIST_PATH = os.getenv("PERSIST_PATH")

class NotesChatAgent:
    
    def __init__(self, model_name = MODEL_NAME, persist_path: Optional[str] = PERSIST_PATH, device = -1, task = "text-generation"):
        
        self.model_name = model_name
        self.device = device
        self.task = task
        self.persist_path = persist_path
        self.notes: Dict[str, str] = {}
        
        self.lf = get_client()
        
        self._load_if_exists()
        self._load_model()
            
    def _load_model(self):
        try:
            print(f"Загружаем модель '{self.model_name}' через pipeline...")

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            self.generator = pipeline(
                self.task,
                model=model,
                tokenizer=tokenizer,
                device=self.device,
                do_sample=False
            )

            self._model_available = True
            print("Модель успешно загружена и готова к работе!")
        except Exception as e:
            print("Ошибка загрузки модели:", e)
            self.generator = None
            self._model_available = False
    
    def list_notes(self) -> Dict[str, str]:
        return dict(self.notes)
    
    def add_note(self, key: str, text: str):
        self.notes[key] = text
        self._autosave()
        
    def update_note(self, key: str, text: str) -> bool:
        if key in self.notes:
            self.notes[key] = text
            self._autosave()
            return True
        return False
    
    def remove_note(self, key: str):
        if key in self.notes:
            del self.notes[key]
            self._autosave()
            return True
        return False
        
    def _autosave(self):
        try:
            self.save()
        except Exception:
            pass
        
    def save(self):
        path = self.persist_path
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.notes, f, ensure_ascii=False, indent=2)
            
    # Сейчас не используется, но должен быть вызван при изменении заметок
    def load(self):
        path = self.persist_path
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.notes = json.load(f)
                
    def _load_if_exists(self):
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'r', encoding='utf-8') as f:
                    self.notes = json.load(f)
            except Exception:
                self.notes = {}
                
    def _compose_context(self, question: str, max_chars: int = None) -> str:
        items = [f"- {k}: {v}" for k, v in self.notes.items()]
        context = "\n".join(items)
        if max_chars is not None:
            if len(context) > max_chars:
                trimmed = []
                cur_len = 0
                for s in items:
                    if cur_len + len(s) > max_chars:
                        break
                    trimmed.append(s)
                    cur_len += len(s)
                context = "\n".join(trimmed)
        prompt = (
            f"Вы — персональный помощник."
            f"Используйте только эти заметки для ответа на вопрос. "
            f"Если информации нет — честно скажите, что не знаете.\n\n"
            f"Контекст: {context}\n\n"
            f"Вопрос: {question}\n"
            f"Ответьте кратко."
        )
        return prompt
    
    def _lifebuoy(self, question: str) -> str:
        q = question.lower()
        for k, v in self.notes.items():
            if any(w in q for w in k.lower().split()):
                return f"По заметке '{k}': {v}"
        return "Я не нашел ответ в заметках. Могу сохранить как новую, если Вы не против."
    
    @observe(name="notes-agent.query", as_type="span")
    def query(self, question: str, max_new_tokens = 150) -> str:
        if not self.notes:
            return "У вас еще нет заметок."
        
        prompt = self._compose_context(question=question)
        if self._model_available and self.generator is not None:
            try:
                start_time = datetime.datetime.now(datetime.UTC)
                
                with self.lf.start_as_current_observation(
                    as_type="generation",
                    name="notes-agent.model-generation",
                    input={"prompt": prompt},
                    metadata={
                        "model": str(self.model_name) if self.model_name else None,
                        "generation_start_time": start_time.isoformat()
                    }
                ) as generation:
                    out = self.generator(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        do_sample=False
                    )
                    if isinstance(out, list) and isinstance(out[0], dict):
                        text = out[0].get('generated_text') or out[0].get('text') or str(out[0])
                    else:
                        text = str(out)

                    generation.update(output={"text": text})
                    
                return text.strip()
        
            except Exception as e:
                try:
                    with self.lf.start_as_current_observation(
                        as_type="span",
                        name="notes-agent.query-error",
                        metadata={"error": str(e)}
                    ) as err_obs:
                        err_obs.update(output=None)
                except Exception:
                    pass
                
                traceback.print_exc()
                return self._lifebuoy(question=question)
        else:
            return self._lifebuoy(question=question)
        
        
if __name__ == '__main__':
    
    print("Запуск demo NotesChatAgent...")
    agent = NotesChatAgent()
    
    agent.add_note('планы', 'Завтра: учеба до 19:00, волейбол после учебы в 20:00')
    agent.add_note('покупки', 'Молоко, хлеб, лапшу, колу, лук')
    agent.add_note('идеи', 'Написать статью о локальных LLM и их приватности')
    
    questions = [
        "Какие у меня планы на завтра?",
        "Что мне нужно купить?",
        "Какие были идеи?"
    ]
    
    for q in questions:
        print('\nQ:', q)
        print('A:', agent.query(q))

    try:
        lf = get_client()
        lf.flush()
        lf.shutdown()
        
        print("Langfuse: flushed and shutdown completed.")
    except Exception as e:
        print("Langfuse flush/shutdown error:", e)