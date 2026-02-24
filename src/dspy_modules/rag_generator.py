import os
import json
import logging
import dspy
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.langchain_integration import RAGPipeline
from .signatures import PlaywrightTestGenerationSignature

logger = logging.getLogger(__name__)

class RAGPlaywrightGenerator(dspy.Module):
    def __init__(self, region: str = "ap-northeast-2"):
        super().__init__()
        # 1. RAG 엔진 초기화
        self.rag_pipeline = RAGPipeline(
            index_name="playwright_code_index",
            region=region
        )
        # 2. DSPy 추론 모듈 설정
        self.chain_of_thought = dspy.ChainOfThought(PlaywrightTestGenerationSignature)
        
        # 3. 결과 저장 경로 설정
        self.save_base_path = "/home/ec2-user/AI/output_codes"
        if not os.path.exists(self.save_base_path):
            os.makedirs(self.save_base_path)
            logger.info(f"Created output directory: {self.save_base_path}")

    def index_documents(self, documents: List[Any], file_tree: Optional[str] = None):
        """저장소의 코드들을 RAG 벡터 DB에 등록"""
        try:
            ids = self.rag_pipeline.add_documents(documents)
            self.file_tree = file_tree
            logger.info(f"Successfully indexed {len(ids)} documents.")
        except Exception as e:
            logger.error(f"Indexing failed: {e}")

    def generate_test(self, requirement: str, base_url: str) -> Dict[str, Any]:
        """코드를 분석/수정하고 로컬 파일로 저장"""
        try:
            # STEP 1: 관련 코드 맥락 검색 (RAG)
            logger.info(f"Searching context for: {requirement}")
            code_context = self.rag_pipeline.retrieve_context(
                requirement=requirement,
                top_k=5,
                file_tree=getattr(self, 'file_tree', None)
            )

            # STEP 2: AI에게 수정 요청 (Bedrock)
            prediction = self.chain_of_thought(
                requirement=requirement,
                code_context=code_context,
                base_url=base_url
            )

            # STEP 3: 파일 저장 처리
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 파일명에서 특수문자 제거
            safe_name = "".join([c for c in requirement[:15] if c.isalnum() or c in ' _-']).strip()
            filename = f"fixed_{timestamp}_{safe_name}.py"
            full_path = os.path.join(self.save_base_path, filename)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(prediction.test_code)

            # STEP 4: 결과 정리
            return {
                "status": "success",
                "saved_file": full_path,
                "test_code": prediction.test_code,
                "reasoning": prediction.reasoning,
                "test_description": prediction.test_description
            }

        except Exception as e:
            logger.error(f"Generation process failed: {e}")
            return {"status": "error", "message": str(e)}

    def clear_index(self):
        """인덱스 초기화"""
        self.rag_pipeline.clear()
        logger.info("RAG Index cleared.")