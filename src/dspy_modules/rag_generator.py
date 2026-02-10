"""
RAG 통합 Playwright 테스트 생성기

RAG 검색 + DSPy 생성을 결합한 모듈입니다.
"""

import dspy
from typing import Dict, Any, Optional, List
import logging

from .signatures import PlaywrightTestGenerationSignature
from src.langchain_integration import RAGPipeline

logger = logging.getLogger(__name__)


class RAGPlaywrightGenerator(dspy.Module):
    """
    RAG 기반 Playwright 테스트 생성기
    
    1. 요구사항을 받아 RAG로 관련 코드 검색
    2. 검색된 코드를 컨텍스트로 DSPy 생성
    3. Playwright 테스트 코드 반환
    
    Attributes:
        rag_pipeline: RAG 파이프라인
        chain_of_thought: DSPy ChainOfThought 모듈
    
    Example:
        >>> from src.dspy_modules import configure_bedrock_dspy, RAGPlaywrightGenerator
        >>> 
        >>> # 1. DSPy 초기화 (Bedrock)
        >>> configure_bedrock_dspy(region="us-east-1")
        >>> 
        >>> # 2. RAG Generator 생성
        >>> generator = RAGPlaywrightGenerator()
        >>> 
        >>> # 3. 코드 인덱싱
        >>> generator.index_repository("/path/to/repo")
        >>> 
        >>> # 4. 테스트 생성
        >>> result = generator.generate_test(
        ...     requirement="사용자 로그인 기능 E2E 테스트",
        ...     base_url="https://example.com"
        ... )
        >>> print(result["test_code"])
    """
    
    def __init__(
        self,
        rag_pipeline: Optional[RAGPipeline] = None,
        region: str = "us-east-1"
    ):
        """
        RAGPlaywrightGenerator 초기화
        
        Args:
            rag_pipeline: 기존 RAG 파이프라인 (없으면 새로 생성)
            region: AWS 리전
        """
        super().__init__()
        
        # RAG 파이프라인 초기화
        if rag_pipeline:
            self.rag_pipeline = rag_pipeline
        else:
            self.rag_pipeline = RAGPipeline(
                index_name="playwright_code_index",
                region=region
            )
        
        # ChainOfThought 생성
        self.chain_of_thought = dspy.ChainOfThought(
            PlaywrightTestGenerationSignature
        )
        
        logger.info("RAGPlaywrightGenerator initialized")
    
    def index_documents(
        self,
        documents: List[Any],
        file_tree: Optional[str] = None
    ) -> None:
        """
        코드 문서를 RAG 파이프라인에 인덱싱
        
        Args:
            documents: LangChain Document 객체 리스트
            file_tree: 전체 파일 구조 (Tree-First 전략용, 저장)
        """
        try:
            # 문서 추가
            ids = self.rag_pipeline.add_documents(documents)
            logger.info(f"Indexed {len(ids)} documents")
            
            # 파일 트리 저장 (나중에 사용)
            if file_tree:
                self.file_tree = file_tree
                logger.info("File tree stored for Tree-First strategy")
            else:
                self.file_tree = None
        
        except Exception as e:
            logger.error(f"Failed to index documents: {e}")
            raise
    
    def forward(
        self,
        requirement: str,
        base_url: str,
        top_k: int = 5,
        max_context_chars: int = 4000
    ) -> dspy.Prediction:
        """
        RAG + 생성 파이프라인 실행
        
        Args:
            requirement: 테스트 요구사항
            base_url: 테스트 대상 URL
            top_k: RAG 검색 개수
            max_context_chars: 최대 컨텍스트 길이
        
        Returns:
            dspy.Prediction: 생성 결과
        """
        try:
            logger.info(f"Generating test for: '{requirement}'")
            
            # 1. RAG 검색 (Tree-First 전략)
            code_context = self.rag_pipeline.retrieve_context(
                requirement=requirement,
                top_k=top_k,
                max_chars=max_context_chars,
                file_tree=getattr(self, 'file_tree', None)
            )
            
            logger.debug(f"Retrieved context: {len(code_context)} chars")
            
            # 2. DSPy 생성
            prediction = self.chain_of_thought(
                requirement=requirement,
                code_context=code_context,
                base_url=base_url
            )
            
            logger.info("Test generation complete")
            
            return prediction
        
        except Exception as e:
            logger.error(f"RAG generation failed: {e}")
            raise
    
    def generate_test(
        self,
        requirement: str,
        base_url: str,
        top_k: int = 5,
        max_context_chars: int = 4000
    ) -> Dict[str, Any]:
        """
        테스트 코드를 딕셔너리로 반환 (API 응답용)
        
        Args:
            requirement: 테스트 요구사항
            base_url: 테스트 대상 URL
            top_k: RAG 검색 개수
            max_context_chars: 최대 컨텍스트 길이
        
        Returns:
            Dict: 테스트 코드 및 메타데이터
                - test_code: str
                - test_description: str
                - test_cases: List[str]
                - lines_of_code: int
                - context_used: int (사용된 컨텍스트 문자 수)
        
        Example:
            >>> generator = RAGPlaywrightGenerator()
            >>> result = generator.generate_test(
            ...     requirement="로그인 테스트",
            ...     base_url="https://example.com"
            ... )
            >>> print(result["test_code"])
        """
        # Forward 호출
        prediction = self.forward(
            requirement=requirement,
            base_url=base_url,
            top_k=top_k,
            max_context_chars=max_context_chars
        )
        
        # test_cases JSON 파싱
        import json
        try:
            test_cases = json.loads(prediction.test_cases)
            if not isinstance(test_cases, list):
                test_cases = [prediction.test_cases]
        except (json.JSONDecodeError, AttributeError):
            logger.warning(f"Failed to parse test_cases as JSON")
            test_cases = [case.strip() for case in prediction.test_cases.split(',')]
        
        # 코드 라인 수 계산
        lines_of_code = len(prediction.test_code.split('\n'))
        
        # 컨텍스트 사용량 (마지막 검색 결과)
        last_context = self.rag_pipeline.retrieve_context(
            requirement=requirement,
            top_k=1,
            max_chars=100
        )
        
        return {
            "test_code": prediction.test_code,
            "test_description": prediction.test_description,
            "test_cases": test_cases,
            "lines_of_code": lines_of_code,
            "rag_top_k": top_k,
        }
    
    def clear_index(self) -> None:
        """RAG 인덱스 초기화"""
        self.rag_pipeline.clear()
        self.file_tree = None
        logger.info("RAG index cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        rag_stats = self.rag_pipeline.get_stats()
        
        return {
            **rag_stats,
            "has_file_tree": hasattr(self, 'file_tree') and self.file_tree is not None
        }


# 사용 예시
if __name__ == "__main__":
    import logging
    from langchain_core.documents import Document
    from .bedrock_lm import BedrockLM
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        print("\n=== Test 1: Initialize DSPy with Bedrock ===")
        bedrock_lm = BedrockLM(region="us-east-1")
        dspy.settings.configure(lm=bedrock_lm)
        print("DSPy configured")
        
        print("\n=== Test 2: Initialize RAG Generator ===")
        generator = RAGPlaywrightGenerator(region="us-east-1")
        print("Generator initialized")
        
        print("\n=== Test 3: Index Sample Code ===")
        sample_docs = [
            Document(
                page_content="""
def login(username, password):
    '''사용자 로그인 함수'''
    if username == "admin" and password == "1234":
        session['user'] = username
        return True
    return False
""",
                metadata={"source": "src/auth/login.py", "language": "python"}
            ),
            Document(
                page_content="""
def logout(user_id):
    '''사용자 로그아웃 함수'''
    if 'user' in session:
        del session['user']
    return True
""",
                metadata={"source": "src/auth/logout.py", "language": "python"}
            ),
        ]
        
        file_tree = """
src/
  auth/
    login.py
    logout.py
  models/
    user.py
"""
        
        generator.index_documents(sample_docs, file_tree=file_tree)
        print("Documents indexed")
        
        print("\n=== Test 4: Generate Test ===")
        result = generator.generate_test(
            requirement="사용자 로그인 기능 E2E 테스트",
            base_url="https://example.com",
            top_k=2
        )
        
        print(f"\nTest Description: {result['test_description']}")
        print(f"Test Cases: {result['test_cases']}")
        print(f"Lines of Code: {result['lines_of_code']}")
        print(f"\nTest Code Preview:")
        print(result['test_code'][:300] + "...")
        
        print("\n=== Test 5: Get Stats ===")
        stats = generator.get_stats()
        print(f"Stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()