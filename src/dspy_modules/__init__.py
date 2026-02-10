"""
DSPy 모듈

이 패키지는 DSPy와 AWS Bedrock을 사용한 AI 기반 테스트 코드 생성을 제공합니다.

주요 컴포넌트:
- BedrockLM: DSPy용 Bedrock 래퍼
- RAGPlaywrightGenerator: RAG 통합 테스트 생성기
- configure_bedrock_dspy: Bedrock 초기화

사용 예시:
    >>> from src.dspy_modules import (
    ...     configure_bedrock_dspy,
    ...     RAGPlaywrightGenerator
    ... )
    >>> 
    >>> # 1. DSPy 초기화 (앱 시작 시 1회)
    >>> configure_bedrock_dspy(region="us-east-1")
    >>> 
    >>> # 2. Generator 생성
    >>> generator = RAGPlaywrightGenerator()
    >>> 
    >>> # 3. 코드 인덱싱
    >>> generator.index_documents(documents, file_tree)
    >>> 
    >>> # 4. 테스트 생성
    >>> result = generator.generate_test(
    ...     requirement="로그인 기능 테스트",
    ...     base_url="https://example.com"
    ... )
"""

from .signatures import PlaywrightTestGenerationSignature
from .bedrock_lm import BedrockLM
from .rag_generator import RAGPlaywrightGenerator

# configure 함수만 dspy_modules에서 가져옴
from .dspy_modules import configure_bedrock_dspy, get_current_lm_info

__all__ = [
    # Signatures
    "PlaywrightTestGenerationSignature",
    
    # Modules
    "RAGPlaywrightGenerator",
    
    # Configuration
    "configure_bedrock_dspy",
    "get_current_lm_info",
    
    # LM
    "BedrockLM",
]

# 버전 정보
__version__ = "2.0.0"