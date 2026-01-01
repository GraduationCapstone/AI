#!/usr/bin/env python3
"""
PROBE - Playwright Test Generator

GitHub 코드를 분석하여 Playwright 테스트 코드를 자동으로 생성하는 시스템

실행 방법:
    python main.py
    
    또는
    
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

환경 변수:
    - ANTHROPIC_API_KEY: Claude API 키
    - OLLAMA_BASE_URL: Ollama 서버 URL
    - POSTGRES_HOST: PostgreSQL 호스트
    - AWS_ACCESS_KEY_ID: S3/MinIO Access Key
    - 기타 설정은 .env 파일 참조
"""

import sys
import os
import logging

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import uvicorn
    from config import settings
    from src.api.main import app
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n필요한 패키지를 설치하세요:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """시작 배너 출력"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗ ██████╗  ██████╗ ██████╗ ███████╗                ║
║   ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗██╔════╝                ║
║   ██████╔╝██████╔╝██║   ██║██████╔╝█████╗                  ║
║   ██╔═══╝ ██╔══██╗██║   ██║██╔══██╗██╔══╝                  ║
║   ██║     ██║  ██║╚██████╔╝██████╔╝███████╗                ║
║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝                ║
║                                                              ║
║              Playwright Test Generator                      ║
║                                                              ║
║   GitHub 코드 → AI 분석 → Playwright 테스트 자동 생성         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_dependencies():
    """필수 의존성 확인"""
    logger.info("Checking dependencies...")
    
    errors = []
    warnings = []
    
    # 1. PostgreSQL 연결 확인
    try:
        from src.database import DatabaseManager
        db = DatabaseManager(settings.database_url)
        stats = db.get_stats()
        logger.info(f"✅ PostgreSQL connected: {stats}")
        db.close()
    except Exception as e:
        errors.append(f"PostgreSQL connection failed: {e}")
    
    # 2. AI 모델 확인
    if settings.is_gpt_enabled:
        try:
            from src.gpt import GPTClient
            gpt = GPTClient(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model
            )
            logger.info(f"✅ GPT (Ollama) accessible: {settings.ollama_base_url}")
        except Exception as e:
            warnings.append(f"GPT (Ollama) connection failed: {e}")
    
    if settings.is_claude_enabled:
        if not settings.anthropic_api_key:
            warnings.append("Claude API key not set")
        else:
            logger.info("✅ Claude API key configured")
    
    # 3. S3/MinIO 확인
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        try:
            from src.storage import FileManager
            fm = FileManager(
                endpoint_url=settings.s3_endpoint_url,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                bucket_name=settings.s3_bucket_name
            )
            logger.info(f"✅ Storage configured: {settings.s3_bucket_name}")
        except Exception as e:
            warnings.append(f"Storage connection failed: {e}")
    else:
        warnings.append("S3/MinIO credentials not configured")
    
    # 결과 출력
    if errors:
        logger.error("\n❌ Critical Errors:")
        for error in errors:
            logger.error(f"  - {error}")
        logger.error("\nPlease fix these errors before starting the server.")
        return False
    
    if warnings:
        logger.warning("\n⚠️ Warnings:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
        logger.warning("\nThe server will start, but some features may be limited.\n")
    
    return True


def main():
    """메인 실행 함수"""
    print_banner()
    
    logger.info("=" * 60)
    logger.info("Starting PROBE Server")
    logger.info("=" * 60)
    
    # 설정 출력
    logger.info(f"\n[Configuration]")
    logger.info(f"Host: {settings.fastapi_host}")
    logger.info(f"Port: {settings.fastapi_port}")
    logger.info(f"AI Model Mode: {settings.ai_model_mode}")
    logger.info(f"Database: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    logger.info(f"Log Level: {settings.log_level}\n")
    
    # 의존성 확인
    if not check_dependencies():
        logger.error("\n❌ Dependency check failed. Exiting.")
        sys.exit(1)
    
    # 서버 실행
    try:
        logger.info("=" * 60)
        logger.info("🚀 Starting FastAPI server...")
        logger.info("=" * 60)
        logger.info(f"\n📖 API Documentation: http://{settings.fastapi_host}:{settings.fastapi_port}/docs")
        logger.info(f"📊 Health Check: http://{settings.fastapi_host}:{settings.fastapi_port}/api/health\n")
        
        uvicorn.run(
            "src.api.main:app",
            host=settings.fastapi_host,
            port=settings.fastapi_port,
            reload=True,  # 개발 모드 (프로덕션에서는 False)
            log_level=settings.log_level.lower(),
            access_log=True
        )
    
    except KeyboardInterrupt:
        logger.info("\n\n👋 Server stopped by user (Ctrl+C)")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"\n\n❌ Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()