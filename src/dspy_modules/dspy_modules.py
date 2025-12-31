"""
DSPy 분석 모듈

Signature를 사용하여 실제 코드 분석을 수행하는 DSPy 모듈입니다.
"""

import dspy
from typing import Dict, Any, Optional
import logging

from .signature import (
    ExecutabilitySignature,
    CodeAnalysisSignature,
    TestGenerationSignature
)

logger = logging.getLogger(__name__)


class PlaywrightTestGenerator(dspy.Module):
    """
    Playwright 테스트 코드 생성 모듈
    
    요구사항과 코드 컨텍스트를 입력받아,
    실행 가능한 Playwright 테스트 코드를 생성합니다.
    
    Attributes:
        chain_of_thought: ChainOfThought 모듈 (단계적 사고)
    
    Example:
        >>> from config import settings
        >>> configure_dspy(settings.ollama_base_url, settings.ollama_model)
        >>> generator = PlaywrightTestGenerator()
        >>> result = generator.generate_to_dict(
        ...     requirement="로그인 기능 E2E 테스트",
        ...     code_context="def login()...",
        ...     base_url="https://example.com"
        ... )
        >>> print(result["test_code"])  # Playwright 코드
    """
    
    def __init__(self):
        """PlaywrightTestGenerator 초기화"""
        super().__init__()
        
        # ChainOfThought 사용 (AI가 단계적으로 사고)
        self.chain_of_thought = dspy.ChainOfThought(PlaywrightTestGenerationSignature)
        
        logger.info("PlaywrightTestGenerator initialized with ChainOfThought")
    
    def forward(
        self,
        requirement: str,
        code_context: str,
        base_url: str
    ) -> dspy.Prediction:
        """
        Playwright 테스트 코드 생성
        
        Args:
            requirement: 테스트 요구사항
            code_context: RAG에서 검색된 코드 컨텍스트
            base_url: 테스트 대상 URL
        
        Returns:
            dspy.Prediction: 생성 결과
                - test_code: str (완전한 Playwright 코드)
                - test_description: str (테스트 설명)
                - test_cases: str (JSON 배열)
        
        Example:
            >>> generator = PlaywrightTestGenerator()
            >>> result = generator(
            ...     requirement="로그인 테스트",
            ...     code_context="def login()...",
            ...     base_url="https://example.com"
            ... )
            >>> print(result.test_code)
        """
        try:
            logger.debug(f"Generating test for: '{requirement[:50]}...'")
            
            # ChainOfThought로 테스트 코드 생성
            prediction = self.chain_of_thought(
                requirement=requirement,
                code_context=code_context,
                base_url=base_url
            )
            
            logger.info(
                f"Test generation complete: {len(prediction.test_code)} chars"
            )
            
            return prediction
        
        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            raise
    
    def generate_to_dict(
        self,
        requirement: str,
        code_context: str,
        base_url: str
    ) -> Dict[str, Any]:
        """
        테스트 코드를 딕셔너리로 반환 (API 응답용)
        
        Args:
            requirement: 테스트 요구사항
            code_context: 코드 컨텍스트
            base_url: 테스트 대상 URL
        
        Returns:
            Dict: JSON 직렬화 가능한 딕셔너리
                - test_code: str (Playwright 코드)
                - test_description: str
                - test_cases: List[str]
                - lines_of_code: int
        
        Example:
            >>> result = generator.generate_to_dict(
            ...     requirement="로그인 테스트",
            ...     code_context="...",
            ...     base_url="https://example.com"
            ... )
            >>> print(result["test_cases"])  # ["정상 로그인", "실패 케이스"]
        """
        prediction = self.forward(requirement, code_context, base_url)
        
        # test_cases JSON 파싱
        import json
        try:
            test_cases = json.loads(prediction.test_cases)
            if not isinstance(test_cases, list):
                test_cases = [prediction.test_cases]
        except (json.JSONDecodeError, AttributeError):
            # JSON 파싱 실패 시 텍스트 분할
            logger.warning(f"Failed to parse test_cases as JSON: {prediction.test_cases}")
            test_cases = [case.strip() for case in prediction.test_cases.split(',')]
        
        # 코드 라인 수 계산
        lines_of_code = len(prediction.test_code.split('\n'))
        
        return {
            "test_code": prediction.test_code,
            "test_description": prediction.test_description,
            "test_cases": test_cases,
            "lines_of_code": lines_of_code
        }


class CodeAnalyzer(dspy.Module):
    """
    일반적인 코드 분석 모듈 (선택사항)
    
    코드 품질, 복잡도, 개선 제안 등을 분석합니다.
    """
    
    def __init__(self):
        super().__init__()
        self.chain_of_thought = dspy.ChainOfThought(CodeAnalysisSignature)
        logger.info("CodeAnalyzer initialized")
    
    def forward(
        self,
        requirement: str,
        code_context: str
    ) -> dspy.Prediction:
        """코드 분석"""
        return self.chain_of_thought(
            requirement=requirement,
            code_context=code_context
        )


class TestGenerator(dspy.Module):
    """
    테스트 케이스 생성 모듈 (미래 확장용)
    
    주어진 코드에 대한 단위 테스트를 자동 생성합니다.
    """
    
    def __init__(self):
        super().__init__()
        self.chain_of_thought = dspy.ChainOfThought(TestGenerationSignature)
        logger.info("TestGenerator initialized")
    
    def forward(
        self,
        code_context: str,
        language: str = "python"
    ) -> dspy.Prediction:
        """테스트 케이스 생성"""
        return self.chain_of_thought(
            code_context=code_context,
            language=language
        )


def configure_dspy(
    base_url: str = "http://localhost:11434",
    model: str = "llama3.1:8b",
    timeout_s: int = 60
) -> None:
    """
    DSPy 설정 (Ollama 연동)
    
    애플리케이션 시작 시 한 번만 호출하면 됩니다.
    
    Args:
        base_url: Ollama 서버 URL
        model: 사용할 Llama 모델
        timeout_s: 요청 타임아웃 (초)
    
    Example:
        >>> from config import settings
        >>> configure_dspy(
        ...     base_url=settings.ollama_base_url,
        ...     model=settings.ollama_model
        ... )
    """
    try:
        # Ollama LM 초기화
        ollama_lm = dspy.OllamaLocal(
            model=model,
            base_url=base_url,
            timeout_s=timeout_s
        )
        
        # DSPy 설정
        dspy.settings.configure(lm=ollama_lm)
        
        logger.info(
            f"DSPy configured with Ollama: "
            f"base_url={base_url}, model={model}"
        )
        
        # 연결 테스트
        test_lm = dspy.settings.lm
        logger.info(f"DSPy LM configured: {test_lm}")
    
    except Exception as e:
        logger.error(f"Failed to configure DSPy: {e}")
        raise ConnectionError(
            f"Cannot connect to Ollama at {base_url}. "
            f"Please ensure Ollama is running (ollama serve) "
            f"and the model '{model}' is installed (ollama pull {model})."
        ) from e


def get_current_lm_info() -> Dict[str, Any]:
    """
    현재 설정된 LM 정보 반환
    
    Returns:
        Dict: LM 정보
    """
    try:
        lm = dspy.settings.lm
        return {
            "configured": True,
            "model": str(lm),
            "type": type(lm).__name__
        }
    except AttributeError:
        return {
            "configured": False,
            "error": "DSPy LM not configured. Call configure_dspy() first."
        }


# 사용 예시
if __name__ == "__main__":
    import logging
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Test 1: Configure DSPy ===")
    configure_dspy(
        base_url="http://localhost:11434",
        model="llama3.1:8b"
    )
    
    print("\n=== Test 2: Check LM Info ===")
    lm_info = get_current_lm_info()
    print(f"LM Info: {lm_info}")
    
    print("\n=== Test 3: ExecutabilityAnalyzer ===")
    analyzer = ExecutabilityAnalyzer()
    
    # 샘플 코드
    sample_code = """
def login(username, password):
    '''사용자 로그인 함수'''
    if username == "admin" and password == "1234":
        return True
    return False

def logout(user_id):
    '''로그아웃 함수'''
    print(f"User {user_id} logged out")
    return True
"""
    
    # 분석
    result = analyzer.analyze_to_dict(
        requirement="사용자 로그인 기능 테스트",
        code_context=sample_code
    )
    
    print(f"\nAnalysis Result:")
    print(f"  Is Executable: {result['is_executable']}")
    print(f"  Reasoning: {result['reasoning']}")
    print(f"  Confidence: {result['confidence_score']}%")
    
    print("\n=== Test 4: Prediction Object ===")
    prediction = analyzer(
        requirement="사용자 로그아웃 기능",
        code_context=sample_code
    )
    
    print(f"  Prediction Type: {type(prediction)}")
    print(f"  Is Executable: {prediction.is_executable}")
    print(f"  Reasoning: {prediction.reasoning[:100]}...")