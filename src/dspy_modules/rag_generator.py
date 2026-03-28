import os
import json
import logging
import dspy
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
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

    def generate_test(self, requirement: str, **kwargs) -> Dict[str, Any]:
        """
        2단계 파이프라인:
          STEP 1 — 레포 분석 → 테스트 계획서 생성 (Excel 저장)
          STEP 2 — 테스트 계획서 → Playwright 테스트 코드 생성 (.spec.js 저장)
        """
        top_k = kwargs.get("top_k", 5)
        execution_id = kwargs.get("execution_id", 0)

        try:
            # ── STEP 1: RAG 컨텍스트 검색 ──────────────────────────────────
            logger.info(f"[{execution_id}] STEP 1 — Retrieving context...")
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
            )

            test_plan_raw = getattr(plan_prediction, "test_plan", "[]")
            plan_reasoning = getattr(plan_prediction, "reasoning", "")

            # test_plan JSON 파싱
            test_cases: List[Dict] = []
            if isinstance(test_plan_raw, str):
                try:
                    cleaned = test_plan_raw.strip()
                    if cleaned.startswith("```"):
                        cleaned = "\n".join(cleaned.split("\n")[1:])
                    if cleaned.endswith("```"):
                        cleaned = "\n".join(cleaned.split("\n")[:-1])
                    test_cases = json.loads(cleaned.strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"[{execution_id}] test_plan JSON parse failed: {e}")
                    logger.warning(f"[{execution_id}] Raw response: {str(test_plan_raw)[:500]}")
                    test_cases = []
            elif isinstance(test_plan_raw, list):
                test_cases = test_plan_raw

            logger.info(f"[{execution_id}] STEP 1 — Generated {len(test_cases)} test cases.")

            # ── STEP 1: 계획서 Excel 저장 ──────────────────────────────────
            timestamp = datetime.now().strftime("%Y%m%d")
            plan_filename = f"{timestamp}_{execution_id}_plan.xlsx"
            plan_path = os.path.join(self.save_base_path, plan_filename)
            _save_plan_as_excel(test_cases, plan_path)
            logger.info(f"[{execution_id}] Saved test plan: {plan_path}")

            # ── STEP 2: Playwright 코드 생성 ───────────────────────────────
            logger.info(f"[{execution_id}] STEP 2 — Generating Playwright test code...")
            test_plan_str = json.dumps(test_cases, ensure_ascii=False, indent=2)

            code_prediction = self.code_generator(
                test_plan=test_plan_str,
                code_context=code_context,
            )

            test_code = getattr(code_prediction, "test_code", "")

            cleaned_code = test_code.strip()
            if cleaned_code.startswith("```"):
                cleaned_code = "\n".join(cleaned_code.split("\n")[1:])
            if cleaned_code.endswith("```"):
                cleaned_code = "\n".join(cleaned_code.split("\n")[:-1])
            test_code = cleaned_code.strip()
            
            code_reasoning = getattr(code_prediction, "reasoning", "")
            logger.info(f"[{execution_id}] STEP 2 — Test code generated ({len(test_code)} chars).")

            # ── STEP 2: 코드 .spec.js 저장 ─────────────────────────────────
            code_filename = f"{timestamp}_{execution_id}.spec.js"
            code_path = os.path.join(self.save_base_path, code_filename)
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(test_code)
            logger.info(f"[{execution_id}] Saved test code: {code_path}")

            return {
                "status": "success",
                "saved_file": code_path,
                "saved_plan": plan_path,
                "test_code": test_code,
                "test_cases": test_cases,
                "plan_reasoning": plan_reasoning,
                "code_reasoning": code_reasoning,
            }

        except Exception as e:
            logger.error(f"[{execution_id}] Generation process failed: {e}")
            return {"status": "error", "message": str(e)}

    # ── 유틸 ────────────────────────────────────────────────────────────────


    def generate_plan_only(self, requirement: str, **kwargs) -> Dict[str, Any]:
        """STEP 1만: 테스트 계획서(Excel) 생성"""
        top_k = kwargs.get("top_k", 5)
        execution_id = kwargs.get("execution_id", 0)

        try:
            logger.info(f"[{execution_id}] STEP 1 — Retrieving context...")
            code_context = self.rag_pipeline.retrieve_context(
                requirement=requirement,
                top_k=top_k,
                max_chars=12000,
                file_tree=getattr(self, "file_tree", None)
            )

            logger.info(f"[{execution_id}] STEP 1 — Generating test plan...")
            plan_prediction = self.plan_generator(
                requirement=requirement,
                code_context=code_context,
            )

            test_plan_raw = getattr(plan_prediction, "test_plan", "[]")
            plan_reasoning = getattr(plan_prediction, "reasoning", "")

            test_cases = []
            if isinstance(test_plan_raw, str):
                try:
                    cleaned = test_plan_raw.strip()
                    if cleaned.startswith("```"):
                        cleaned = "\n".join(cleaned.split("\n")[1:])
                    if cleaned.endswith("```"):
                        cleaned = "\n".join(cleaned.split("\n")[:-1])
                    test_cases = json.loads(cleaned.strip())
                except json.JSONDecodeError as e:
                    logger.warning(f"[{execution_id}] test_plan JSON parse failed: {e}")
                    logger.warning(f"[{execution_id}] Raw response: {str(test_plan_raw)[:500]}")
                    test_cases = []
            elif isinstance(test_plan_raw, list):
                test_cases = test_plan_raw

            logger.info(f"[{execution_id}] STEP 1 — Generated {len(test_cases)} test cases.")

            timestamp = datetime.now().strftime("%Y%m%d")
            plan_filename = f"{timestamp}_{execution_id}_plan.xlsx"
            plan_path = os.path.join(self.save_base_path, plan_filename)
            _save_plan_as_excel(test_cases, plan_path)
            logger.info(f"[{execution_id}] Saved test plan: {plan_path}")

            return {
                "status": "success",
                "saved_plan": plan_path,
                "test_cases": test_cases,
                "plan_reasoning": plan_reasoning,
            }

        except Exception as e:
            logger.error(f"[{execution_id}] Plan generation failed: {e}")
            return {"status": "error", "message": str(e)}

    def generate_code_only(self, requirement: str, **kwargs) -> Dict[str, Any]:
        """STEP 2만: 기존 계획서 기반 테스트 코드 생성"""
        top_k = kwargs.get("top_k", 5)
        execution_id = kwargs.get("execution_id", 0)
        plan_path = kwargs.get("plan_path", None)

        try:
            logger.info(f"[{execution_id}] STEP 2 — Retrieving context...")
            code_context = self.rag_pipeline.retrieve_context(
                requirement=requirement,
                top_k=top_k,
                max_chars=12000,
                file_tree=getattr(self, "file_tree", None)
            )

            # 계획서 로드 (있으면)
            test_cases = []
            if plan_path and os.path.exists(plan_path):
                try:
                    import openpyxl
                    wb = openpyxl.load_workbook(plan_path)
                    ws = wb.active
                    for row in ws.iter_rows(min_row=6, values_only=True):
                        if row[4]:  # case_id 컬럼
                            test_cases.append({
                                "no": row[0],
                                "scenario_id": row[1],
                                "scenario_name": row[2],
                                "description": row[3],
                                "case_id": row[4],
                                "case_name": row[5],
                                "precondition": row[6],
                                "test_data": row[7],
                                "steps": row[8],
                                "expected_result": row[9],
                            })
                except Exception as e:
                    logger.warning(f"[{execution_id}] Failed to load plan: {e}")

            logger.info(f"[{execution_id}] STEP 2 — Generating test code from {len(test_cases)} cases...")
            test_plan_str = json.dumps(test_cases, ensure_ascii=False, indent=2)

            code_prediction = self.code_generator(
                test_plan=test_plan_str,
                code_context=code_context,
            )

            test_code = getattr(code_prediction, "test_code", "")
            # 마크다운 코드 블록 제거
            cleaned = test_code.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
            if cleaned.endswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[:-1])
            test_code = cleaned.strip()

            logger.info(f"[{execution_id}] STEP 2 — Code generated ({len(test_code)} chars).")

            timestamp = datetime.now().strftime("%Y%m%d")
            code_filename = f"{timestamp}_{execution_id}.spec.js"
            code_path = os.path.join(self.save_base_path, code_filename)
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(test_code)
            logger.info(f"[{execution_id}] Saved test code: {code_path}")

            return {
                "status": "success",
                "saved_file": code_path,
                "saved_plan": plan_path,
                "test_code": test_code,
                "test_cases": test_cases,
            }

        except Exception as e:
            logger.error(f"[{execution_id}] Code generation failed: {e}")
            return {"status": "error", "message": str(e)}

    def clear_index(self):
        self.rag_pipeline.clear()
        logger.info("RAG Index cleared.")


# ── Excel 저장 함수 ───────────────────────────────────────────────────────────


def _save_plan_as_excel(test_cases: List[Dict], path: str) -> None:
    """SIT 시나리오 양식에 맞게 테스트 계획서 Excel 저장"""
    from openpyxl.utils import get_column_letter
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "SIT시나리오"

    thin = Side(style="thin", color="000000")
    border = Border(top=thin, bottom=thin, left=thin, right=thin)
    header_font = Font(name="Arial", bold=True, size=10)
    header_fill = PatternFill(start_color="BDD7EE", end_color="BDD7EE", fill_type="solid")
    title_fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")
    title_font = Font(name="Arial", bold=True, size=12, color="FFFFFF")
    center = Alignment(horizontal="center", vertical="center", wrap_text=True)
    left = Alignment(horizontal="left", vertical="center", wrap_text=True)
    data_font = Font(name="Arial", size=10)

    # 1행: 타이틀
    ws.merge_cells("A1:N1")
    ws["A1"] = "SIT 시나리오"
    ws["A1"].font = title_font
    ws["A1"].fill = title_fill
    ws["A1"].alignment = center
    ws["A1"].border = border
    ws.row_dimensions[1].height = 22

    # 4~5행: 헤더
    for merge_range, value in [
        ("A4:A5", "No"), ("B4:B5", "테스트시나리오ID"),
        ("C4:C5", "테스트시나리오명"), ("D4:D5", "설명"),
        ("E4:J4", "테스트케이스"), ("K4:N4", "테스트 결과"),
    ]:
        ws.merge_cells(merge_range)
        cell = ws[merge_range.split(":")[0]]
        cell.value = value
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border

    for cell_ref, value in [
        ("E5", "테스트케이스ID"), ("F5", "테스트케이스명"),
        ("G5", "전제조건"), ("H5", "테스트 데이터"),
        ("I5", "실행단계"), ("J5", "예상결과"),
        ("K5", "테스트결과\n(화면)"), ("L5", "결과\n(정상/결함)"),
        ("M5", "테스트일자\n(YYYY-MM-DD)"), ("N5", "테스터"),
    ]:
        cell = ws[cell_ref]
        cell.value = value
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border

    ws.row_dimensions[4].height = 22
    ws.row_dimensions[5].height = 30

    # 시나리오별 그룹핑
    groups = []
    current_group = []
    for tc in test_cases:
        if tc.get("no") is not None:
            if current_group:
                groups.append(current_group)
            current_group = [tc]
        else:
            current_group.append(tc)
    if current_group:
        groups.append(current_group)

    row_idx = 6
    for group in groups:
        group_start = row_idx
        group_end = row_idx + len(group) - 1

        for i, tc in enumerate(group):
            if i == 0:
                ws.cell(row=row_idx, column=1, value=tc.get("no", ""))
                ws.cell(row=row_idx, column=2, value=tc.get("scenario_id", ""))
                ws.cell(row=row_idx, column=3, value=tc.get("scenario_name", ""))
                ws.cell(row=row_idx, column=4, value=tc.get("description", ""))

            for col, field in [
                (5, "case_id"), (6, "case_name"), (7, "precondition"),
                (8, "test_data"), (9, "steps"), (10, "expected_result")
            ]:
                val = tc.get(field, "")
                if isinstance(val, (list, dict)):
                    val = json.dumps(val, ensure_ascii=False)
                ws.cell(row=row_idx, column=col, value=str(val) if val else "")

            for col in range(1, 15):
                cell = ws.cell(row=row_idx, column=col)
                cell.font = data_font
                cell.border = border
                cell.alignment = center if col in [1, 2, 3, 5, 7] else left

            ws.row_dimensions[row_idx].height = 55
            row_idx += 1

        if group_end > group_start:
            for col in [1, 2, 3, 4]:
                ws.merge_cells(
                    start_row=group_start, start_column=col,
                    end_row=group_end, end_column=col
                )
                ws.cell(row=group_start, column=col).alignment = center

    for col, width in zip("ABCDEFGHIJKLMN", [7,18,18,50,16,20,16,25,35,25,17,11,15,11]):
        ws.column_dimensions[col].width = width

    ws.freeze_panes = "A6"
    wb.save(path)
