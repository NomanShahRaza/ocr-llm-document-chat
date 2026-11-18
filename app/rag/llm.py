from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (Path(__file__).parent / 'prompts' / 'answer_system.md').read_text(encoding='utf-8')
CITATION_REMINDER = (
    'Answer the question directly in synthesized prose. '
    'Do not include citations, file names, or page references such as [{example} p.3] in the answer.'
)


class AnswerLLM:
    def __init__(self) -> None:
        self.model = LLM_MODEL
        self.base_url = LLM_BASE_URL
        self._client = None
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return bool(self.model)

    def _get_client(self):
        if self._client is None:
            with self._lock:
                if self._client is None:
                    from openai import OpenAI

                    # Local servers (Ollama, LM Studio) accept any non-empty key.
                    self._client = OpenAI(
                        base_url=self.base_url,
                        api_key=LLM_API_KEY or 'not-needed',
                        timeout=LLM_TIMEOUT,
                    )
        return self._client

    def answer(
        self,
        question: str,
        context: str,
        question_type: str | None = None,
        history: list[dict[str, str]] | None = None,
        citation_file: str = 'document.pdf',
    ) -> str:
        reminder = CITATION_REMINDER.format(example=citation_file)
        client = self._get_client()
        hint = f'Detected question type: {question_type}\n\n' if question_type else ''
        # Prior turns are kept short so the retrieved evidence still fits the context window.
        prior = [
            {'role': turn['role'], 'content': turn['content'][:1200]}
            for turn in (history or [])
            if turn.get('role') in {'user', 'assistant'}
        ]
        response = client.chat.completions.create(
            model=self.model,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                *prior,
                {'role': 'user', 'content': f'User question:\n{question}\n\n{hint}{context}\n\n{reminder}'},
            ],
        )
        text = (response.choices[0].message.content or '').strip()
        if not text:
            raise RuntimeError('LLM returned an empty answer')
        return text


answer_llm = AnswerLLM()
