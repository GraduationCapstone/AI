"""
API 모듈

FastAPI를 사용한 RESTful API 서버입니다.

주요 엔드포인트:
- POST /api/generate-test: Playwright 테스트 코드 생성
- GET /api/health: 서버 상태 확인
- GET /api/tests/{script_id}: 테스트 스크립트 조회
"""

from .main import app

__all__ = ["app"]