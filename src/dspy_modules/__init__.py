"""
DSPy 모듈

이 패키지는 DSPy를 사용한 AI 기반 코드 분석을 제공합니다.

주요 컴포넌트:
- ExecutabilitySignature: 입출력 정의
- ExecutabilityAnalyzer: 실행 가능성 분석
- configure_dspy: DSPy 초기화 (Ollama 연동)
- ExecutabilityOptimizer: 프롬프트 최적화 (선택사항)

사용 예시:
    >>> from config import settings
    >>> from src.dspy_modules import configure_dspy, ExecutabilityAnalyzer
    >>> 
    >>> # 1. DSPy 초기화 (앱 시작 시 1회)
    >>> configure_dspy(
    ...     base_url=settings.ollama_base_url,
    ...     model=settings.ollama_model
    ... )
    >>> 
    >>> # 2. Analyzer 생성
    >>> analyzer = ExecutabilityAnalyzer()
    >>> 
    >>> # 3. 코드 분석
    >>> result = analyzer.analyze_to_dict(
    ...     requirement="사용자 로그인 기능 테스트",
    ...     code_context="def login()..."
    ... )
    >>> 
    >>> print(result)
    {
        "is_executable": True,
        "reasoning": "...",
        "confidence_score": 85
    }

참고:
- DSPy는 프롬프트를 자동으로 생성하므로, 프롬프트 작성이 불필요합니다.
- ChainOfThought를 사용하여 AI가 단계적으로 사고합니다.
- Optimizer는 학습 데이터가 있을 때 사용하는 고급 기능입니다.
"""

from .signature import (
    ExecutabilitySignature,
    CodeAnalysisSignature,
    TestGenerationSignature,
)

from .dspy_modules import (
    ExecutabilityAnalyzer,
    CodeAnalyzer,
    TestGenerator,
    configure_dspy,
    get_current_lm_info,
)

from .optimizer import (
    ExecutabilityOptimizer,
    OptimizerConfig,
)

__all__ = [
    # Signatures
    "ExecutabilitySignature",
    "CodeAnalysisSignature",
    "TestGenerationSignature",
    
    # Modules
    "ExecutabilityAnalyzer",
    "CodeAnalyzer",
    "TestGenerator",
    
    # Configuration
    "configure_dspy",
    "get_current_lm_info",
    
    # Optimizer
    "ExecutabilityOptimizer",
    "OptimizerConfig",
]

# 버전 정보
__version__ = "1.0.0"

# 패키지 레벨 독스트링
__doc__ = """
DSPy 모듈 - AI 기반 코드 분석

이 패키지는 DSPy 프레임워크를 사용하여 코드 실행 가능성을 자동으로 분석합니다.

핵심 기능:
1. 자동 프롬프트 생성 (프롬프트 작성 불필요)
2. Chain of Thought (단계적 사고)
3. Ollama 로컬 LLM 연동
4. 프롬프트 최적화 (선택사항)

사용 흐름:
configure_dspy() → ExecutabilityAnalyzer() → analyze_to_dict()

최소 사용 예시:
    from src.dspy_modules import configure_dspy, ExecutabilityAnalyzer
    
    configure_dspy("http://localhost:11434", "llama3.1:8b")
    analyzer = ExecutabilityAnalyzer()
    
    result = analyzer.analyze_to_dict(
        requirement="로그인 기능",
        code_context="def login(): ..."
    )
    
    print(result["is_executable"])  # True/False
"""