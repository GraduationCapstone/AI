"""
PROBE AI Server - 메인 실행 파일

FastAPI 서버를 실행합니다.

실행 방법:
    # 개발 모드
    python main.py

    # 프로덕션 모드
    python main.py --production

    # 커스텀 포트
    python main.py --port 8080

    # 호스트 지정
    python main.py --host 127.0.0.1 --port 8080
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import uvicorn
from config import settings, validate_settings, print_settings


def setup_logging(log_level: str = "INFO"):
    """로깅 설정"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/app.log', encoding='utf-8')
        ]
    )


def parse_args():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='PROBE AI Server - Playwright 테스트 자동 생성',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 개발 모드 (기본)
  python main.py

  # 프로덕션 모드
  python main.py --production

  # 커스텀 설정
  python main.py --host 0.0.0.0 --port 8080 --workers 4

  # 설정 확인만
  python main.py --check-config
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help=f'서버 호스트 (기본값: {settings.fastapi_host})'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=None,
        help=f'서버 포트 (기본값: {settings.fastapi_port})'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='워커 프로세스 수 (기본값: 1, 프로덕션: 4 권장)'
    )
    
    parser.add_argument(
        '--production',
        action='store_true',
        help='프로덕션 모드로 실행 (자동 리로드 비활성화)'
    )
    
    parser.add_argument(
        '--reload',
        action='store_true',
        help='자동 리로드 활성화 (개발 모드에서만 사용)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default=None,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help=f'로그 레벨 (기본값: {settings.log_level})'
    )
    
    parser.add_argument(
        '--check-config',
        action='store_true',
        help='설정 확인만 하고 종료'
    )
    
    parser.add_argument(
        '--print-config',
        action='store_true',
        help='설정 출력'
    )
    
    return parser.parse_args()


def check_dependencies():
    """필수 패키지 확인"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'boto3',
        'dspy',
        'langchain',
        'faiss'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ 누락된 패키지: {', '.join(missing_packages)}")
        print("\n설치 방법:")
        print("  pip install -r requirements.txt\n")
        sys.exit(1)
    
    print("✅ 모든 필수 패키지가 설치되어 있습니다.")


def create_directories():
    """필요한 디렉토리 생성"""
    directories = ['logs', 'temp']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print(f"✅ 디렉토리 생성 완료: {', '.join(directories)}")


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    # 로그 레벨 설정
    log_level = args.log_level or settings.log_level
    
    # 로그 디렉토리 생성
    create_directories()
    
    # 로깅 설정
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 60)
    print("🚀 PROBE AI Server")
    print("   Playwright 테스트 자동 생성 시스템")
    print("=" * 60 + "\n")
    
    # 설정 출력 요청
    if args.print_config:
        print_settings(mask_secrets=True)
        return
    
    # 패키지 확인
    print("📦 필수 패키지 확인 중...")
    check_dependencies()
    
    # 설정 검증
    print("\n⚙️  설정 검증 중...")
    try:
        validate_settings()
        print("✅ 설정 검증 완료")
    except ValueError as e:
        print(f"\n❌ 설정 오류:\n{e}\n")
        print("💡 .env 파일을 확인하거나 환경 변수를 설정하세요.")
        print("   예시: cp .env.example .env\n")
        sys.exit(1)
    
    # 설정 확인만 하고 종료
    if args.check_config:
        print("\n✅ 설정이 올바릅니다!")
        print_settings(mask_secrets=True)
        return
    
    # 서버 설정
    host = args.host or settings.fastapi_host
    port = args.port or settings.fastapi_port
    workers = args.workers
    
    # 개발/프로덕션 모드 결정
    if args.production:
        reload = False
        if workers == 1:
            workers = 4  # 프로덕션은 기본 4 워커
        logger.info("🏭 프로덕션 모드로 실행")
    else:
        reload = args.reload or True  # 개발 모드는 기본 자동 리로드
        workers = 1  # 개발 모드는 단일 워커
        logger.info("🔧 개발 모드로 실행 (자동 리로드 활성화)")
    
    # 서버 정보 출력
    print(f"\n📡 서버 정보:")
    print(f"   - 주소: http://{host}:{port}")
    print(f"   - 워커 수: {workers}")
    print(f"   - 자동 리로드: {'✅' if reload else '❌'}")
    print(f"   - 로그 레벨: {log_level}")
    print(f"   - 환경: {settings.environment}")
    print(f"   - AWS 리전: {settings.aws_region}")
    print(f"   - Bedrock 모델: {settings.bedrock_model}")
    
    # API 문서 안내
    print(f"\n📚 API 문서:")
    print(f"   - Swagger UI: http://{host}:{port}/docs")
    print(f"   - ReDoc: http://{host}:{port}/redoc")
    
    # Health Check 안내
    print(f"\n💚 Health Check:")
    print(f"   curl http://{host}:{port}/api/health")
    
    print("\n" + "=" * 60)
    print("✨ 서버를 시작합니다...\n")
    
    # 서버 실행
    try:
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=log_level.lower(),
            access_log=True,
            use_colors=True
        )
    except KeyboardInterrupt:
        print("\n\n👋 서버를 종료합니다...")
        logger.info("서버 종료")
    except Exception as e:
        logger.error(f"서버 실행 중 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
