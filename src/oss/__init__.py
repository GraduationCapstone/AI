"""
GPT OSS (Ollama) 모듈

이 모듈은 Ollama를 통해 로컬에서 실행되는 GPT OSS 모델과의 통신을 담당합니다.
"""

from .client import GPTClient

__all__ = ["GPTClient"]