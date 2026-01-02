"""
PROBE - Playwright Test Generator

이 패키지는 GitHub 코드를 분석하여 Playwright 테스트 코드를 자동으로 생성합니다.

모듈 구조:
    - api: FastAPI 서버 (REST API)
    - gpt: GPT OSS 클라이언트 (Ollama)
    - claude: Claude API 클라이언트 (Anthropic)
    - dspy_modules: DSPy 프롬프트 최적화
    - storage: S3/MinIO 파일 저장
    - database: PostgreSQL + pgvector
    - embeddings: 벡터 임베딩 생성
    - langchain_integration: RAG 파이프라인

사용 예시:
    >>> from src.api import app
    >>> from src.database import DatabaseManager
    >>> from src.dspy_modules import PlaywrightTestGenerator
    >>> 
    >>> # FastAPI 앱 실행
    >>> import uvicorn
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)
"""

__version__ = "1.0.0"
__author__ = "PROBE Team"
__description__ = "Playwright Test Generator using AI"

# 주요 컴포넌트 노출
from .api import app
from .database import DatabaseManager, Repository, CodeChunk, TestScript
from .dspy_modules import PlaywrightTestGenerator, configure_dspy
from .embeddings import get_embedding_generator
from .langchain_integration import RAGPipeline
from .storage import FileManager
from .gpt import GPTClient
from .claude import ClaudeClient

__all__ = [
    # FastAPI
    "app",
    
    # Database
    "DatabaseManager",
    "Repository",
    "CodeChunk",
    "TestScript",
    
    # AI
    "PlaywrightTestGenerator",
    "configure_dspy",
    "GPTClient",
    "ClaudeClient",
    
    # Utilities
    "get_embedding_generator",
    "RAGPipeline",
    "FileManager",
]