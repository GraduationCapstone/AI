"""
Configuration 모듈

환경 변수 기반 설정을 관리합니다.

사용 예시:
    >>> from config import settings, validate_settings, print_settings
    >>> 
    >>> # 설정 출력
    >>> print_settings()
    >>> 
    >>> # 설정 검증
    >>> validate_settings()
    >>> 
    >>> # 설정 사용
    >>> print(settings.database_url)
    >>> print(settings.ai_model_mode)
"""

from .config import (
    Settings,
    settings,
    validate_settings,
    print_settings,
)

__all__ = [
    "Settings",
    "settings",
    "validate_settings",
    "print_settings",
]

__version__ = "1.0.0"