"""
애플리케이션 설정 관리 (Dual AI Model)

GPT OSS (Ollama) + Claude API 이중 모델 지원
"""

import os
from typing import Optional, Literal
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()


class Settings(BaseSettings):
    """
    애플리케이션 설정 (Dual AI Model)
    
    지원하는 AI 모델:
    1. Claude API (Anthropic) - 외부 API, 높은 품질
    2. GPT OSS (Ollama) - 로컬 실행, 무료
    """
    
    # ===== AI Models =====
    # Claude API (Anthropic)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")
    
    # GPT OSS (Ollama)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gpt-3.5-turbo")
    
    # AI 모드 선택
    ai_model_mode: Literal["claude", "gpt", "both"] = os.getenv("AI_MODEL_MODE", "both")
    
    # ===== GitHub =====
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    
    # ===== PostgreSQL =====
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "probe_db")
    postgres_user: str = os.getenv("POSTGRES_USER", "probe_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    
    # ===== S3/MinIO Storage =====
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    s3_bucket_name: str = os.getenv("S3_BUCKET_NAME", "probe-tests")
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    
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
        if not self.ollama_base_url.endswith("/v1"):
            return f"{self.ollama_base_url}/v1"
        return self.ollama_base_url
    
    @property
    def is_claude_enabled(self) -> bool:
        """Claude API 사용 가능 여부"""
        return bool(self.anthropic_api_key) and self.ai_model_mode in ["claude", "both"]
    
    @property
    def is_gpt_enabled(self) -> bool:
        """GPT OSS 사용 가능 여부"""
        return self.ai_model_mode in ["gpt", "both"]
    
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
    """
    errors = []
    
    # GitHub Token 검증
    if not settings.github_token:
        errors.append("GITHUB_TOKEN is not set (required for GitHub API)")
    
    # PostgreSQL Password 검증
    if not settings.postgres_password:
        errors.append("POSTGRES_PASSWORD is not set (required for database connection)")
    
    # AI 모델 검증
    if settings.ai_model_mode == "claude" or settings.ai_model_mode == "both":
        if not settings.anthropic_api_key:
            errors.append(
                "ANTHROPIC_API_KEY is not set but AI_MODEL_MODE includes 'claude'"
            )
    
    if settings.ai_model_mode == "gpt" or settings.ai_model_mode == "both":
        if not settings.ollama_base_url:
            errors.append("OLLAMA_BASE_URL is not set")
        if not settings.ollama_model:
            errors.append("OLLAMA_MODEL is not set")
    
    # S3/MinIO 검증
    if not settings.aws_access_key_id:
        errors.append("AWS_ACCESS_KEY_ID is not set (required for S3/MinIO)")
    if not settings.aws_secret_access_key:
        errors.append("AWS_SECRET_ACCESS_KEY is not set (required for S3/MinIO)")
    
    if errors:
        error_message = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_message)
        raise ValueError(error_message)
    
    logger.info("Configuration validation successful")
    logger.info(f"AI Model Mode: {settings.ai_model_mode}")
    logger.info(f"  - Claude: {settings.is_claude_enabled}")
    logger.info(f"  - GPT OSS: {settings.is_gpt_enabled}")
    
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
    
    print("\n=== PROBE Configuration (Dual AI Model) ===")
    print("\n[AI Models]")
    print(f"AI Model Mode: {settings.ai_model_mode}")
    print(f"  Claude Enabled: {settings.is_claude_enabled}")
    print(f"  GPT Enabled: {settings.is_gpt_enabled}")
    
    if settings.is_claude_enabled:
        print(f"\n[Claude API]")
        print(f"API Key: {mask(settings.anthropic_api_key) if mask_secrets else settings.anthropic_api_key}")
        print(f"Model: {settings.claude_model}")
    
    if settings.is_gpt_enabled:
        print(f"\n[GPT OSS (Ollama)]")
        print(f"Base URL: {settings.ollama_base_url}")
        print(f"Model: {settings.ollama_model}")
    
    print(f"\n[GitHub]")
    print(f"Token: {mask(settings.github_token) if mask_secrets else settings.github_token}")
    
    print(f"\n[PostgreSQL]")
    print(f"Host: {settings.postgres_host}:{settings.postgres_port}")
    print(f"Database: {settings.postgres_db}")
    print(f"User: {settings.postgres_user}")
    print(f"Password: {mask(settings.postgres_password) if mask_secrets else settings.postgres_password}")
    
    print(f"\n[S3/MinIO Storage]")
    print(f"Endpoint: {settings.s3_endpoint_url}")
    print(f"Bucket: {settings.s3_bucket_name}")
    print(f"Access Key: {mask(settings.aws_access_key_id) if mask_secrets else settings.aws_access_key_id}")
    print(f"Secret Key: {mask(settings.aws_secret_access_key) if mask_secrets else settings.aws_secret_access_key}")
    print(f"Region: {settings.aws_region}")
    
    print(f"\n[FastAPI]")
    print(f"Host: {settings.fastapi_host}:{settings.fastapi_port}")
    
    print(f"\n[Other]")
    print(f"Embedding Model: {settings.embedding_model}")
    print(f"Log Level: {settings.log_level}")
    print(f"Database URL: {mask(settings.database_url) if mask_secrets else settings.database_url}")
    print("=" * 50 + "\n")


# 모듈 로드 시 자동 검증 (선택 사항)
if __name__ == "__main__":
    try:
        validate_settings()
        print_settings(mask_secrets=True)
    except ValueError as e:
        print(f"Error: {e}")