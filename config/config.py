"""
애플리케이션 설정 관리 (AWS Bedrock)

AWS Bedrock 기반 설정
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
    애플리케이션 설정 (AWS Bedrock)
    
    AI 모델: AWS Bedrock (Claude 3.5 Sonnet)
    """
    
    # ===== AWS Bedrock =====
    aws_region: str = os.getenv("AWS_REGION", "us-east-1")
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    
    # Bedrock 모델
    bedrock_model: str = os.getenv(
        "BEDROCK_MODEL",
        "anthropic.claude-3-5-sonnet-20241022-v2:0"
    )
    
    # Bedrock Embeddings
    bedrock_embedding_model: str = os.getenv(
        "BEDROCK_EMBEDDING_MODEL",
        "amazon.titan-embed-text-v2:0"
    )
    bedrock_embedding_dimension: int = int(os.getenv("BEDROCK_EMBEDDING_DIMENSION", "1024"))
    
    # ===== GitHub =====
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    
    # ===== FastAPI =====
    fastapi_host: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    fastapi_port: int = int(os.getenv("FASTAPI_PORT", "8000"))
    
    # ===== S3 Storage (결과 저장용) =====
    s3_bucket_name: str = os.getenv("S3_BUCKET_NAME", "probe-tests")
    s3_results_prefix: str = os.getenv("S3_RESULTS_PREFIX", "test-results/")
    
    # ===== Spring Boot Webhook =====
    springboot_webhook_url: str = os.getenv(
        "SPRINGBOOT_WEBHOOK_URL",
        "http://localhost:8080/api/webhook/test-complete"
    )
    
    # ===== RAG 설정 =====
    rag_chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    rag_chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    rag_max_context_chars: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000"))
    
    # ===== Logging =====
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # ===== 개발/프로덕션 환경 =====
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    @property
    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.environment.lower() == "development"
    
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
    
    # AWS 자격증명 검증 (개발 환경에서는 IAM Role 사용 가능)
    if settings.is_production:
        if not settings.aws_access_key_id:
            errors.append("AWS_ACCESS_KEY_ID is not set (required for production)")
        if not settings.aws_secret_access_key:
            errors.append("AWS_SECRET_ACCESS_KEY is not set (required for production)")
    
    # GitHub Token 검증 (선택사항, public repo는 불필요)
    if not settings.github_token:
        logger.warning("GITHUB_TOKEN is not set (required for private repositories)")
    
    # S3 Bucket 검증
    if not settings.s3_bucket_name:
        errors.append("S3_BUCKET_NAME is not set (required for test result storage)")
    
    if errors:
        error_message = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_message)
        raise ValueError(error_message)
    
    logger.info("Configuration validation successful")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"AWS Region: {settings.aws_region}")
    logger.info(f"Bedrock Model: {settings.bedrock_model}")
    
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
    
    print("\n=== PROBE Configuration (AWS Bedrock) ===")
    print(f"\n[Environment]")
    print(f"Mode: {settings.environment}")
    print(f"Production: {settings.is_production}")
    
    print(f"\n[AWS Bedrock]")
    print(f"Region: {settings.aws_region}")
    print(f"Model: {settings.bedrock_model}")
    print(f"Embedding Model: {settings.bedrock_embedding_model}")
    print(f"Embedding Dimension: {settings.bedrock_embedding_dimension}")
    
    if settings.aws_access_key_id:
        print(f"Access Key: {mask(settings.aws_access_key_id) if mask_secrets else settings.aws_access_key_id}")
        print(f"Secret Key: {mask(settings.aws_secret_access_key) if mask_secrets else settings.aws_secret_access_key}")
    else:
        print(f"Credentials: Using IAM Role")
    
    print(f"\n[GitHub]")
    if settings.github_token:
        print(f"Token: {mask(settings.github_token) if mask_secrets else settings.github_token}")
    else:
        print(f"Token: Not set (public repos only)")
    
    print(f"\n[S3 Storage]")
    print(f"Bucket: {settings.s3_bucket_name}")
    print(f"Results Prefix: {settings.s3_results_prefix}")
    
    print(f"\n[FastAPI]")
    print(f"Host: {settings.fastapi_host}:{settings.fastapi_port}")
    
    print(f"\n[Spring Boot]")
    print(f"Webhook URL: {settings.springboot_webhook_url}")
    
    print(f"\n[RAG Settings]")
    print(f"Chunk Size: {settings.rag_chunk_size}")
    print(f"Chunk Overlap: {settings.rag_chunk_overlap}")
    print(f"Top K: {settings.rag_top_k}")
    print(f"Max Context: {settings.rag_max_context_chars} chars")
    
    print(f"\n[Other]")
    print(f"Log Level: {settings.log_level}")
    print("=" * 50 + "\n")


# 모듈 로드 시 자동 검증 (선택 사항)
if __name__ == "__main__":
    try:
        validate_settings()
        print_settings(mask_secrets=True)
    except ValueError as e:
        print(f"Error: {e}")