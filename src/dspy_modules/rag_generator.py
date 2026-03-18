import os
import json
import logging
import dspy
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.langchain_integration import RAGPipeline
from .signatures import TestPlanGenerationSignature, TestCodeGenerationSignature

logger = logging.getLogger(__name__)


class RAGPlaywrightGenerator(dspy.Module):
    def __init__(self, region: str = "ap-northeast-1"):
        super().__init__()

        # 1. RAG 엔진 초기화
        self.rag_pipeline = RAGPipeline(
            index_name="playwright_code_index",
            region=region
        )

        # 2. DSPy 추론 모듈 — 2단계 파이프라인
        self.plan_generator = dspy.ChainOfThought(TestPlanGenerationSignature)
        self.code_generator = dspy.ChainOfThought(TestCodeGenerationSignature)

        # 3. 결과 저장 경로
        self.save_base_path = "/home/ec2-user/AI/output_codes"
        if not os.path.exists(self.save_base_path):
            os.makedirs(self.save_base_path)
            logger.info(f"Created output directory: {self.save_base_path}")

    # ── 인덱싱 ──────────────────────────────────────────────────────────────

    def index_documents(self, documents: List[Any], file_tree: Optional[str] = None):
        """저장소 코드를 RAG 벡터 DB에 등록"""
        try:
            ids = self.rag_pipeline.add_documents(documents)
            self.file_tree = file_tree
            logger.info(f"Successfully indexed {len(ids)} documents.")
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise

    # ── 메인 생성 로직 ───────────────────────────────────────────────────────

    def generate_test(self, requirement: str, base_url: str, **kwargs) -> Dict[str, Any]:
        """
        2단계 파이프라인:
          STEP 1 — 테스트 계획서 생성 (TestPlanGenerationSignature)
          STEP 2 — Playwright 테스트 코드 생성 (TestCodeGenerationSignature)
        """
        top_k = kwargs.get("top_k", 5)
        execution_id = kwargs.get("execution_id", 0)

        try:
            # ── STEP 1: RAG 컨텍스트 검색 ──────────────────────────────────
            logger.info(f"[{execution_id}] STEP 1 — Retrieving context for: {requirement[:80]}")
            code_context = self.rag_pipeline.retrieve_context(
                requirement=requirement,
                top_k=top_k,
                max_chars=12000,
                file_tree=getattr(self, "file_tree", None)
            )

            # ── STEP 1: 테스트 계획서 생성 ─────────────────────────────────
            logger.info(f"[{execution_id}] STEP 1 — Generating test plan...")
            plan_prediction = self.plan_generator(
                requirement=requirement,
                code_context=code_context,
                base_url=base_url
            )

            test_plan_raw = getattr(plan_prediction, "test_plan", "[]")
            plan_reasoning = getattr(plan_prediction, "reasoning", "")

            # test_plan이 문자열이면 JSON 파싱 시도
            test_cases: List[Dict] = []
            if isinstance(test_plan_raw, str):
                try:
                    # 마크다운 코드 블록 제거
                    cleaned = test_plan_raw.strip()
                    if cleaned.startswith("```"):
                        cleaned = "\n".join(cleaned.split("\n")[1:])
                    if cleaned.endswith("```"):
                        cleaned = "\n".join(cleaned.split("\n")[:-1])
                    test_cases = json.loads(cleaned.strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"[{execution_id}] Failed to parse test_plan JSON: {e}. Using raw string.")
                    test_cases = []
            elif isinstance(test_plan_raw, list):
                test_cases = test_plan_raw

            logger.info(f"[{execution_id}] STEP 1 — Generated {len(test_cases)} test cases.")

            # ── STEP 2: 테스트 코드 생성 ───────────────────────────────────
            logger.info(f"[{execution_id}] STEP 2 — Generating Playwright test code...")

            # test_plan을 문자열로 직렬화하여 전달
            test_plan_str = json.dumps(test_cases, ensure_ascii=False, indent=2)

            code_prediction = self.code_generator(
                test_plan=test_plan_str,
                code_context=code_context,
                base_url=base_url
            )

            test_code = getattr(code_prediction, "test_code", "")
            code_reasoning = getattr(code_prediction, "reasoning", "")

            logger.info(f"[{execution_id}] STEP 2 — Test code generated ({len(test_code)} chars).")

            # ── STEP 3: 파일 저장 ──────────────────────────────────────────
            timestamp = datetime.now().strftime("%Y%m%d")
            filename = f"{timestamp}_{execution_id}.spec.js"
            full_path = os.path.join(self.save_base_path, filename)

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(test_code)

            logger.info(f"[{execution_id}] Saved test file: {full_path}")

            return {
                "status": "success",
                "saved_file": full_path,
                "test_code": test_code,
                "test_cases": test_cases,
                "plan_reasoning": plan_reasoning,
                "code_reasoning": code_reasoning,
            }

        except Exception as e:
            logger.error(f"[{execution_id}] Generation process failed: {e}")
            return {"status": "error", "message": str(e)}

    # ── 유틸 ────────────────────────────────────────────────────────────────

    def clear_index(self):
        """인덱스 초기화"""
        self.rag_pipeline.clear()
        logger.info("RAG Index cleared.")