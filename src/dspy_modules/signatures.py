"""
DSPy Signature 정의

Signature는 AI 모델의 입력과 출력을 정의합니다.
프롬프트를 직접 작성하지 않고, DSPy가 자동으로 생성합니다.
"""

import dspy


class PlaywrightTestGenerationSignature(dspy.Signature):
    """
    Playwright 테스트 코드 생성 Signature
    
    입력:
        - requirement: 테스트 요구사항 (예: "사용자 로그인 기능 테스트")
        - code_context: GitHub에서 추출한 관련 코드 (RAG 검색 결과)
        - base_url: 테스트 대상 URL (예: "https://example.com")
    
    출력:
        - test_code: 완전한 Playwright 테스트 코드 (JavaScript)
        - test_description: 테스트 시나리오 설명
        - test_cases: 포함된 테스트 케이스 목록 (JSON 배열)
    """
    
    # ===== 입력 필드 =====
    requirement: str = dspy.InputField(
        desc=(
            "테스트하려는 기능에 대한 설명. "
            "예: '사용자 로그인 기능 테스트', '상품 검색 기능 테스트'"
        )
    )
    
    code_context: str = dspy.InputField(
        desc=(
            "GitHub 저장소에서 추출한 관련 코드. "
            "이 코드를 분석하여 실제 구현된 기능을 파악하고, "
            "그에 맞는 Playwright 테스트를 생성하세요."
        )
    )
    
    base_url: str = dspy.InputField(
        desc=(
            "테스트 대상 애플리케이션의 기본 URL. "
            "예: 'https://example.com' 또는 'http://localhost:3000'"
        )
    )
    
    # ===== 출력 필드 =====
    test_code: str = dspy.OutputField(
        desc=(
            "완전히 실행 가능한 Playwright 테스트 코드를 JavaScript로 작성하세요. "
            "반드시 다음 구조를 따르세요:\n"
            "1. import 문 포함: import { test, expect } from '@playwright/test';\n"
            "2. test.describe() 블록 사용\n"
            "3. 각 테스트 케이스는 test() 함수로 작성\n"
            "4. async/await 패턴 사용\n"
            "5. 실제 CSS 셀렉터 및 URL 경로 포함\n"
            "6. expect() 단언문 포함하여 검증\n"
            "코드만 출력하고, 설명이나 마크다운은 포함하지 마세요."
        )
    )
    
    test_description: str = dspy.OutputField(
        desc=(
            "생성된 테스트의 전체적인 목적과 시나리오를 간단히 설명하세요. "
            "어떤 기능을 테스트하는지, 어떤 경우들을 검증하는지 명시하세요."
        )
    )
    
    test_cases: str = dspy.OutputField(
        desc=(
            "생성된 테스트에 포함된 개별 테스트 케이스들의 목록을 JSON 배열로 작성하세요. "
            "예: [\"정상 로그인\", \"잘못된 비밀번호\", \"빈 필드 검증\"]"
        )
    )
