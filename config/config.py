
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# .env 파일 로드
load_dotenv()


class Settings(BaseSettings):
    """애플리케이션 설정"""
    
    # Claude API
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    
    # GitHub
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    
    # PostgreSQL
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "probe_db")
    postgres_user: str = os.getenv("POSTGRES_USER", "probe_user")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    
    # FastAPI
    fastapi_host: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    fastapi_port: int = int(os.getenv("FASTAPI_PORT", "8000"))

    @property
    def database_url(self) -> str:
        """PostgreSQL 연결 URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# 전역 설정 객체
settings = Settings()


def validate_settings():
    """필수 환경 변수 검증"""
    errors = []
    
    if not settings.anthropic_api_key:
        errors.append("ANTHROPIC_API_KEY is not set")
    
    if not settings.github_token:
        errors.append("GITHUB_TOKEN is not set")
    
    if not settings.postgres_password:
        errors.append("POSTGRES_PASSWORD is not set")
    
    if errors:
        raise ValueError(
            "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        )
    
    return True
