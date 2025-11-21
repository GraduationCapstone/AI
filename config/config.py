"""
애플리케이션 설정 관리

.env 파일에서 환경 변수를 로드하고 검증하는 역할.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()


class Settings(BaseSettings):
    """
    애플리케이션 설정
    
    모든 환경 변수는 .env 파일에서 로드됩니다.
    """
    
    # ===== Ollama 설정 =====
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    
    # ===== GitHub =====
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    
    # ===== PostgreSQL =====
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "probe_db")
    postgres_user: str = os.getenv("POSTGRES_USER", "probe_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    
    # ===== FastAPI =====
    fastapi_host: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    fastapi_port: int = int(os.getenv("FASTAPI_PORT", "8000"))
    
    # ===== Embedding Model =====
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # ===== Logging =====
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def database_url(self) -> str:
        """
        PostgreSQL 연결 URL 생성
        
        Returns:
            str: postgresql://user:password@host:port/database
        """
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def ollama_api_url(self) -> str:
        """
        Ollama API URL (OpenAI 호환)
        
        Returns:
            str: http://localhost:11434/v1
        """
        # /v1이 없으면 추가
        if not self.ollama_base_url.endswith("/v1"):
            return f"{self.ollama_base_url}/v1"
        return self.ollama_base_url
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 전역 설정 객체
settings = Settings()


def validate_settings() -> bool:
    """
    필수 환경 변수 검증
    
    Returns:
        bool: 검증 성공 시 True
    
    Raises:
        ValueError: 필수 환경 변수가 없는 경우
    
    Example:
        >>> from config import validate_settings
        >>> validate_settings()
        True
    """
    errors = []
    
    # GitHub Token 검증
    if not settings.github_token:
        errors.append("GITHUB_TOKEN is not set (required for GitHub API)")
    
    # PostgreSQL Password 검증
    if not settings.postgres_password:
        errors.append("POSTGRES_PASSWORD is not set (required for database connection)")
    
    # Ollama 설정 검증
    if not settings.ollama_base_url:
        errors.append("OLLAMA_BASE_URL is not set")
    
    if not settings.ollama_model:
        errors.append("OLLAMA_MODEL is not set")
    
    if errors:
        error_message = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_message)
        raise ValueError(error_message)
    
    logger.info("Configuration validation successful")
    return True


def get_settings() -> Settings:
    """
    설정 객체 반환 (싱글톤)
    
    Returns:
        Settings: 전역 설정 객체
    """
    return settings


def print_settings(mask_secrets: bool = True) -> None:
    """
    현재 설정 출력 (디버깅용)
    
    Args:
        mask_secrets: True이면 비밀번호/토큰을 마스킹
    """
    def mask(value: str, show_chars: int = 4) -> str:
        """값을 마스킹 처리"""
        if not value or len(value) <= show_chars:
            return "***"
        return value[:show_chars] + "***"
    
    print("\n=== PROBE Configuration ===")
    print(f"Ollama Base URL: {settings.ollama_base_url}")
    print(f"Ollama Model: {settings.ollama_model}")
    print(f"GitHub Token: {mask(settings.github_token) if mask_secrets else settings.github_token}")
    print(f"PostgreSQL Host: {settings.postgres_host}:{settings.postgres_port}")
    print(f"PostgreSQL Database: {settings.postgres_db}")
    print(f"PostgreSQL User: {settings.postgres_user}")
    print(f"PostgreSQL Password: {mask(settings.postgres_password) if mask_secrets else settings.postgres_password}")
    print(f"FastAPI: {settings.fastapi_host}:{settings.fastapi_port}")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Log Level: {settings.log_level}")
    print(f"Database URL: {mask(settings.database_url) if mask_secrets else settings.database_url}")
    print("=" * 30 + "\n")


# 모듈 로드 시 자동 검증 
if __name__ == "__main__":
    try:
        validate_settings()
        print_settings(mask_secrets=True)
    except ValueError as e:
        print(f"Error: {e}")