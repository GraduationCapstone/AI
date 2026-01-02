"""
Storage 모듈

이 모듈은 생성된 Playwright 테스트 스크립트를 S3/MinIO에 저장합니다.
"""

from .file_manager import FileManager

__all__ = ["FileManager"]