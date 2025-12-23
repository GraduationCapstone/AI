"""
DSPy Signature 정의

Signature는 AI 모델의 입력과 출력을 정의합니다.
프롬프트를 직접 작성하지 않고, DSPy가 자동으로 생성합니다.
"""

import dspy
from typing import Optional


class ExecutabilitySignature(dspy.Signature):
    """
    코드 실행 가능성 분석 Signature
    
    입력:
        - requirement: 사용자 요구사항
        - code_context: 분석할 코드 컨텍스트 (RAG에서 검색된 코드)
    
    출력:
        - is_executable: 실행 가능 여부 (true/false)
        - reasoning: 판단 근거 (상세 설명)
        - confidence_score: 신뢰도 점수 (0-100)
    
    Example:
        >>> signature = ExecutabilitySignature
        >>> analyzer = dspy.ChainOfThought(signature)
        >>> result = analyzer(
        ...     requirement="사용자 로그인 기능 테스트",
        ...     code_context="def login(user, pwd): ..."
        ... )
        >>> print(result.is_executable)  # "true" or "false"
    """
    
    # ===== 입력 필드 =====
    requirement: str = dspy.InputField(
        desc="사용자가 요청한 기능 또는 테스트 요구사항"
    )
    
    code_context: str = dspy.InputField(
        desc=(
            "GitHub 저장소에서 추출한 관련 코드 조각들. "
            "RAG 시스템이 요구사항과 유사한 코드를 검색한 결과입니다."
        )
    )
    
    # ===== 출력 필드 =====
    is_executable: str = dspy.OutputField(
        desc=(
            "코드가 요구사항을 만족하는지 여부. "
            "반드시 'true' 또는 'false' 문자열로 답변하세요."
        )
    )
    
    reasoning: str = dspy.OutputField(
        desc=(
            "판단 근거를 상세히 설명하세요. "
            "어떤 코드가 요구사항을 만족하는지, "
            "또는 왜 만족하지 못하는지 구체적으로 작성하세요."
        )
    )
    
    confidence_score: str = dspy.OutputField(
        desc=(
            "분석 결과의 신뢰도를 0에서 100 사이의 정수로 표현하세요. "
            "100은 완전히 확신, 0은 전혀 확신하지 못함을 의미합니다."
        )
    )


class CodeAnalysisSignature(dspy.Signature):
    """
    일반적인 코드 분석 Signature (선택사항)
    
    더 세밀한 분석이 필요한 경우 사용할 수 있습니다.
    """
    
    requirement: str = dspy.InputField(desc="분석 요구사항")
    code_context: str = dspy.InputField(desc="코드 컨텍스트")
    
    analysis: str = dspy.OutputField(desc="코드 분석 결과")
    suggestions: str = dspy.OutputField(desc="개선 제안사항")
    complexity: str = dspy.OutputField(desc="코드 복잡도 (low/medium/high)")


class TestGenerationSignature(dspy.Signature):
    """
    테스트 케이스 생성 Signature (미래 확장용)
    
    코드에 대한 테스트 케이스를 자동 생성할 때 사용합니다.
    """
    
    code_context: str = dspy.InputField(desc="테스트할 코드")
    language: str = dspy.InputField(desc="프로그래밍 언어 (python, javascript 등)")
    
    test_cases: str = dspy.OutputField(desc="생성된 테스트 케이스 코드")
    coverage: str = dspy.OutputField(desc="예상 테스트 커버리지 (%)")


# 사용 예시
if __name__ == "__main__":
    import dspy
    
    # Ollama 설정 (테스트용)
    ollama_lm = dspy.OllamaLocal(
        model="llama3.1:8b",
        base_url="http://localhost:11434"
    )
    dspy.settings.configure(lm=ollama_lm)
    
    # ExecutabilitySignature 사용
    print("\n=== Test 1: ExecutabilitySignature ===")
    analyzer = dspy.ChainOfThought(ExecutabilitySignature)
    
    result = analyzer(
        requirement="사용자 로그인 기능 구현",
        code_context="""
def login(username, password):
    '''사용자 로그인 함수'''
    if username == "admin" and password == "1234":
        return True
    return False
"""
    )
    
    print(f"Is Executable: {result.is_executable}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Confidence: {result.confidence_score}")
    
    # CodeAnalysisSignature 사용
    print("\n=== Test 2: CodeAnalysisSignature ===")
    code_analyzer = dspy.ChainOfThought(CodeAnalysisSignature)
    
    result2 = code_analyzer(
        requirement="코드 품질 분석",
        code_context="def add(a, b): return a + b"
    )
    
    print(f"Analysis: {result2.analysis}")
    print(f"Complexity: {result2.complexity}")