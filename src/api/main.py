"""
FastAPI 서버

Spring Boot와 통신하는 AI 서버입니다.

엔드포인트:
  POST /api/generate-plan  — 레포 분석 → 테스트 계획서(Excel) 생성 → S3 → 콜백
  POST /api/generate-test  — 계획서 기반 테스트 코드 생성 → Playwright 실행 → 엑셀 결과 업데이트 → S3 → 콜백
  GET  /api/result/{execution_id} — 생성 결과 조회
  GET  /api/health — 서버 상태
"""

from fastapi import FastAPI, BackgroundTasks, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict
import asyncio
import logging
import tempfile
import shutil
import subprocess
import os
import re
import json
import time
import httpx
import boto3
from datetime import datetime

from config.config import settings, validate_settings
from src.dspy_modules import configure_bedrock_dspy, RAGPlaywrightGenerator
from src.langchain_integration import CodeChunker
from langchain_core.documents import Document

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PROBE AI Server",
    description="AI 기반 Playwright 테스트 코드 자동 생성 서버",
    version="2.0.0"
)

execution_store: dict = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 모델 ─────────────────────────────────────────────────────────────────────

class GeneratePlanRequest(BaseModel):
    executionId: int 
    targetBranch: str = "main"
    repository_url: Optional[HttpUrl] = "https://github.com/Danimo1/logintest"
    requirement: Optional[str] = "base_url: https://danimo1.github.io/logintest/\n\n이 레포 분석해서 로그인 시나리오 테스트 계획서 작성하고 playwright 테스트 코드 생성"
    auth_token: Optional[str] = "토큰값"
    callback_url: str = "http://10.0.1.243:8080/api/agent/callback"

    @property
    def execution_id(self) -> int:
        return self.executionId

    @property
    def branch(self) -> str:
        return self.targetBranch


class GenerateTestRequest(BaseModel):
    executionId: int 
    targetBranch: str = "main"
    repository_url: Optional[HttpUrl] = "https://github.com/Danimo1/logintest"
    requirement: Optional[str] = "base_url: https://danimo1.github.io/logintest/\n\n이 레포 분석해서 로그인 시나리오 테스트 계획서 작성하고 playwright 테스트 코드 생성"
    auth_token: Optional[str] = "토큰값"
    callback_url: str = "http://10.0.1.243:8080/api/agent/callback"

    @property
    def execution_id(self) -> int:
        return self.executionId

    @property
    def branch(self) -> str:
        return self.targetBranch


class TestCaseResult(BaseModel):
    case_name: Optional[str] = None
    status: str
    error_log: Optional[str] = None
    screenshot_s3_url: Optional[str] = None


class CallbackPayload(BaseModel):
    execution_id: int
    status: str
    duration_ms: Optional[int] = None
    report_s3_url: Optional[str] = None
    results: Optional[List[TestCaseResult]] = None


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────

EXCLUDE_DIRS = {
    ".git", "node_modules", "dist", "build", ".next",
    "__pycache__", "vendor", "target", "public", ".vite"
}

OUTPUT_DIR = "/home/ec2-user/AI/output_codes"


def _clone_sync(repo_url: str, branch: str, token: Optional[str]) -> str:
    tmp_dir = tempfile.mkdtemp()
    url = repo_url
    if token:
        url = url.replace("https://", f"https://{token}@")
    try:
        subprocess.run(
            ["git", "clone", "-b", branch, "--depth", "1", url, tmp_dir],
            check=True, capture_output=True
        )
        return tmp_dir
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp_dir)
        raise Exception(f"Git clone failed: {e.stderr.decode()}")


def _get_file_tree(path: str) -> str:
    tree = []
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        level = root.replace(path, "").count(os.sep)
        tree.append(f"{'    ' * level}{os.path.basename(root)}/")
        for f in files:
            tree.append(f"{'    ' * (level + 1)}{f}")
    return "\n".join(tree)


def _load_documents(repo_path: str) -> List[Document]:
    documents = []
    target_extensions = (".js", ".ts", ".tsx", ".java", ".html", ".css")
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(target_extensions):
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 100 * 1024:
                    logger.warning(f"Skipping large file: {file_path}")
                    continue
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    rel_path = os.path.relpath(file_path, repo_path)
                    documents.append(Document(page_content=content, metadata={"source": rel_path}))
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")
    return documents


async def _clone_and_chunk(repo_url: str, branch: str, auth_token: Optional[str], execution_id: int):
    """클론 → 파일 로드 → 청킹 공통 로직"""
    repo_path = await asyncio.to_thread(_clone_sync, repo_url, branch, auth_token)

    logger.info(f"[{execution_id}] Scanning & loading files...")
    file_tree, documents = await asyncio.gather(
        asyncio.to_thread(_get_file_tree, repo_path),
        asyncio.to_thread(_load_documents, repo_path),
    )

    if not documents:
        shutil.rmtree(repo_path, ignore_errors=True)
        raise ValueError("분석할 소스 코드 파일(.js, .html 등)을 찾을 수 없습니다.")

    logger.info(f"[{execution_id}] Chunking documents...")
    chunker = CodeChunker(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap
    )
    chunked_docs = await asyncio.to_thread(chunker.split_by_file_extension, documents)

    return repo_path, file_tree, chunked_docs


S3_BUCKET = "kiwi-test-artifacts-probe-2026"
S3_REGION = "ap-northeast-1"

def _upload_to_s3(local_path: str, s3_key: str) -> Optional[str]:
    try:
        s3 = boto3.client("s3", region_name=S3_REGION)
        s3.upload_file(local_path, S3_BUCKET, s3_key)
        url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"Uploaded to S3: {url}")
        return url
    except Exception as e:
        logger.warning(f"S3 upload skipped: {e}")
        return None

def _upload_bytes_to_s3(data: bytes, s3_key: str, content_type: str) -> Optional[str]:
    try:
        s3 = boto3.client("s3", region_name=S3_REGION)
        s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=data, ContentType=content_type)
        url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"Uploaded bytes to S3: {url}")
        return url
    except Exception as e:
        logger.warning(f"S3 bytes upload skipped: {e}")
        return None


async def send_callback(callback_url: str, payload: CallbackPayload) -> None:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                callback_url,
                json=payload.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            logger.info(f"Callback OK → {callback_url} (execution_id={payload.execution_id})")
    except Exception as e:
        logger.error(f"Callback FAILED → {callback_url} (execution_id={payload.execution_id}): {e}")


# ── PLAN 전용 함수 ────────────────────────────────────────────────────────────

def _run_plan_pipeline(chunked_docs, file_tree: str, requirement: str, execution_id: int) -> dict:
    """STEP 1만: 테스트 계획서(Excel) 생성"""
    from src.dspy_modules.rag_generator import RAGPlaywrightGenerator as Gen
    generator = Gen(region=settings.aws_region)
    generator.index_documents(chunked_docs, file_tree=file_tree)
    return generator.generate_plan_only(
        requirement=requirement,
        top_k=settings.rag_top_k,
        execution_id=execution_id,
    )


# ── TEST 전용 함수 ────────────────────────────────────────────────────────────

def _run_test_pipeline(chunked_docs, file_tree: str, requirement: str, execution_id: int) -> dict:
    """STEP 2만: 테스트 코드 생성 (계획서는 이미 존재)"""
    from src.dspy_modules.rag_generator import RAGPlaywrightGenerator as Gen
    generator = Gen(region=settings.aws_region)
    generator.index_documents(chunked_docs, file_tree=file_tree)

    # 기존 계획서 로드
    today = datetime.now().strftime("%Y%m%d")
    plan_path = os.path.join(OUTPUT_DIR, f"{today}_{execution_id}_plan.xlsx")

    return generator.generate_code_only(
        requirement=requirement,
        top_k=settings.rag_top_k,
        execution_id=execution_id,
        plan_path=plan_path if os.path.exists(plan_path) else None,
    )


def _run_playwright(spec_path: str, execution_id: int) -> List[Dict]:
    result_dir = tempfile.mkdtemp()
    json_report_path = os.path.join(result_dir, "report.json")

    try:
        # JSON 리포트를 파일로 직접 지정 (--reporter=json 단독으로 쓰면 stdout으로 나옴)
        # --output-file 대신 PLAYWRIGHT_JSON_OUTPUT_NAME 환경변수 사용
        env = {
            **os.environ,
            "CI": "1",
            "PLAYWRIGHT_JSON_OUTPUT_NAME": json_report_path,
        }
        cmd = [
            "npx", "playwright", "test", spec_path,
            "--reporter=json",
            f"--output={result_dir}",
        ]
        proc = subprocess.run(
            cmd,
            timeout=300,
            cwd="/home/ec2-user/AI",
            env=env,
            capture_output=True,
        )
        logger.info(f"[{execution_id}] Playwright exit code: {proc.returncode}")

        # stderr 확인 (exit code 1은 테스트 실패이므로 에러로 처리하지 않음)
        if proc.stderr:
            stderr_preview = proc.stderr.decode(errors="ignore")[:300]
            if stderr_preview.strip():
                logger.warning(f"[{execution_id}] Playwright stderr: {stderr_preview}")

        results = []

        # 1순위: PLAYWRIGHT_JSON_OUTPUT_NAME으로 저장된 파일
        if os.path.exists(json_report_path) and os.path.getsize(json_report_path) > 0:
            try:
                with open(json_report_path) as jf:
                    report = json.load(jf)
                results = _parse_playwright_json(report)
                logger.info(f"[{execution_id}] Parsed {len(results)} results from JSON file")
            except Exception as e:
                logger.warning(f"[{execution_id}] JSON file parse failed: {e}")

        # 2순위: stdout에서 JSON 파싱 (일부 버전은 stdout으로 출력)
        if not results and proc.stdout:
            stdout = proc.stdout.decode(errors="ignore").strip()
            try:
                results = _parse_playwright_json(json.loads(stdout))
                logger.info(f"[{execution_id}] Parsed {len(results)} results from stdout")
            except Exception:
                pass

        # 3순위: 텍스트 파싱 (fallback)
        if not results:
            combined = ""
            if proc.stdout:
                combined += proc.stdout.decode(errors="ignore")
            if proc.stderr:
                combined += proc.stderr.decode(errors="ignore")
            results = _parse_playwright_text(combined, execution_id)

        return results

    except subprocess.TimeoutExpired:
        return [{"case_name": "Playwright Execution", "status": "FAIL",
                 "error_log": "Timeout", "screenshot_path": None}]
    except Exception as e:
        return [{"case_name": "Playwright Execution", "status": "FAIL",
                 "error_log": str(e), "screenshot_path": None}]
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


def _parse_playwright_json(report: dict) -> List[Dict]:
    results = []

    def walk_suite(suite):
        for spec in suite.get("specs", []):
            case_name = spec.get("title", "Unknown")
            ok = spec.get("ok", False)
            error_log = None
            for test in spec.get("tests", []):
                for result in test.get("results", []):
                    if not ok and not error_log:
                        for err in result.get("errors", []):
                            msg = err.get("message", "")
                            error_log = msg[:300] if msg else str(err)
                            break
            results.append({
                "case_name": case_name,
                "status": "SUCCESS" if ok else "FAIL",
                "error_log": error_log,
                "screenshot_path": None,
            })
        for sub in suite.get("suites", []):
            walk_suite(sub)

    for suite in report.get("suites", []):
        walk_suite(suite)

    return results or [{"case_name": "Playwright Execution", "status": "FAIL", "error_log": "No results", "screenshot_path": None}]


def _parse_playwright_text(output: str, execution_id: int) -> List[Dict]:
    results = []
    for match in re.finditer(r'✓\s+\d+\s+(.+?)\s+\(\d+', output):
        results.append({"case_name": match.group(1).strip(), "status": "SUCCESS", "error_log": None, "screenshot_path": None})
    for match in re.finditer(r'✘\s+\d+\s+(.+?)\s+\(\d+', output):
        results.append({"case_name": match.group(1).strip(), "status": "FAIL", "error_log": "Test failed", "screenshot_path": None})
    if not results:
        passed = "passed" in output.lower()
        results.append({"case_name": "Playwright Execution", "status": "SUCCESS" if passed else "FAIL",
                        "error_log": None if passed else output[-500:], "screenshot_path": None})
    return results


def _generate_result_screenshot(case_idx: int, all_results: List[Dict]) -> Optional[str]:
    screenshot_dir = os.path.join(OUTPUT_DIR, "test-results")
    os.makedirs(screenshot_dir, exist_ok=True)
    screenshot_path = os.path.join(screenshot_dir, f"result_{str(case_idx).zfill(3)}.png")

    rows = ""
    for i, r in enumerate(all_results, 1):
        is_current = (i == case_idx)
        bg = "#fff9c4" if is_current else "white"
        border_style = "2px solid #4f46e5" if is_current else "1px solid #e5e7eb"
        passed = "1" if r.get("status") == "SUCCESS" else ""
        failed = "1" if r.get("status") == "FAIL" else ""
        p_color = "#16a34a" if passed else "#9ca3af"
        f_color = "#dc2626" if failed else "#9ca3af"
        fw = "bold" if is_current else "normal"
        rows += f"""<tr style="background:{bg};">
            <td style="padding:8px 12px;text-align:center;font-weight:{fw};">{i}</td>
            <td style="padding:8px 12px;max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:{fw};outline:{border_style};">{r.get('case_name', '')}</td>
            <td style="padding:8px 12px;text-align:center;color:{p_color};font-weight:bold;">{passed}</td>
            <td style="padding:8px 12px;text-align:center;color:{f_color};font-weight:bold;">{failed}</td>
        </tr>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
  body {{ margin:0; padding:20px; font-family:Arial,sans-serif; background:#f4f6f9; }}
  .card {{ background:white; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,.1); overflow:hidden; max-width:560px; margin:0 auto; }}
  .header {{ background:#4f46e5; color:white; padding:14px 20px; font-size:16px; font-weight:bold; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ background:#f8fafc; padding:10px 12px; text-align:left; font-size:12px; color:#6b7280; border-bottom:2px solid #e5e7eb; }}
  td {{ font-size:13px; border-bottom:1px solid #f1f5f9; }}
</style></head><body>
<div class="card">
  <div class="header">Testing Results</div>
  <table><thead><tr>
    <th style="width:40px;text-align:center;">No</th>
    <th>Test Case</th>
    <th style="width:60px;text-align:center;color:#16a34a;">Passed</th>
    <th style="width:60px;text-align:center;color:#dc2626;">Failed</th>
  </tr></thead><tbody>{rows}</tbody></table>
</div></body></html>"""

    html_path = os.path.join(screenshot_dir, f"result_{str(case_idx).zfill(3)}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # ── 버그2 수정: playwright CLI 대신 Node.js 인라인 스크립트 사용 ──
    js_script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.setViewportSize({{ width: 620, height: {max(120, len(all_results) * 36 + 80)} }});
  await page.goto('file://{html_path}');
  await page.waitForTimeout(300);
  await page.screenshot({{ path: '{screenshot_path}', fullPage: true }});
  await browser.close();
}})();
"""
    js_path = os.path.join(screenshot_dir, f"capture_{str(case_idx).zfill(3)}.js")
    try:
        with open(js_path, "w") as f:
            f.write(js_script)
        subprocess.run(
            ["node", js_path],
            capture_output=True, timeout=30,
            cwd="/home/ec2-user/AI"
        )
        return screenshot_path if os.path.exists(screenshot_path) else None
    except Exception as e:
        logger.warning(f"Result screenshot failed: {e}")
        return None
    finally:
        for p in [html_path, js_path]:
            if os.path.exists(p):
                os.remove(p)

def _generate_summary_screenshot(all_results: List[Dict]) -> Optional[str]:
    """전체 결과를 하나의 스크린샷으로 생성"""
    screenshot_dir = os.path.join(OUTPUT_DIR, "test-results")
    os.makedirs(screenshot_dir, exist_ok=True)
    summary_path = os.path.join(screenshot_dir, "summary.png")

    total = len(all_results)
    passed = sum(1 for r in all_results if r.get("status") == "SUCCESS")
    failed = total - passed

    rows = ""
    for i, r in enumerate(all_results, 1):
        status = r.get("status", "FAIL")
        p = "1" if status == "SUCCESS" else ""
        f = "1" if status == "FAIL" else ""
        p_color = "#16a34a" if p else "#9ca3af"
        f_color = "#dc2626" if f else "#9ca3af"
        bg = "#f0fdf4" if status == "SUCCESS" else "#fff1f2"
        rows += f"""<tr style="background:{bg};">
            <td style="padding:7px 12px;text-align:center;">{i}</td>
            <td style="padding:7px 12px;max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{r.get('case_name', '')}</td>
            <td style="padding:7px 12px;text-align:center;color:{p_color};font-weight:bold;">{p}</td>
            <td style="padding:7px 12px;text-align:center;color:{f_color};font-weight:bold;">{f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
  body {{ margin:0; padding:24px; font-family:Arial,sans-serif; background:#f4f6f9; }}
  .card {{ background:white; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,.1); overflow:hidden; max-width:600px; margin:0 auto; }}
  .header {{ background:#4f46e5; color:white; padding:16px 20px; font-size:17px; font-weight:bold; }}
  .summary {{ display:flex; gap:24px; padding:14px 20px; background:#f8fafc; border-bottom:1px solid #e5e7eb; font-size:13px; }}
  .badge {{ padding:4px 12px; border-radius:20px; font-weight:bold; }}
  .pass {{ background:#dcfce7; color:#16a34a; }}
  .fail {{ background:#fee2e2; color:#dc2626; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ background:#f1f5f9; padding:10px 12px; text-align:left; font-size:12px; color:#6b7280; border-bottom:2px solid #e5e7eb; }}
  td {{ font-size:13px; border-bottom:1px solid #f1f5f9; }}
</style></head><body>
<div class="card">
  <div class="header">Testing Results Summary</div>
  <div class="summary">
    <span>Total: <strong>{total}</strong></span>
    <span class="badge pass">Passed: {passed}</span>
    <span class="badge fail">Failed: {failed}</span>
  </div>
  <table><thead><tr>
    <th style="width:40px;text-align:center;">No</th>
    <th>Test Case</th>
    <th style="width:65px;text-align:center;color:#16a34a;">Passed</th>
    <th style="width:65px;text-align:center;color:#dc2626;">Failed</th>
  </tr></thead><tbody>{rows}</tbody></table>
</div></body></html>"""

    html_path = os.path.join(screenshot_dir, "summary.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    js_script = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.setViewportSize({{ width: 648, height: {max(200, total * 36 + 140)} }});
  await page.goto('file://{html_path}');
  await page.waitForTimeout(300);
  await page.screenshot({{ path: '{summary_path}', fullPage: true }});
  await browser.close();
}})();
"""
    js_path = os.path.join(screenshot_dir, "capture_summary.js")
    try:
        with open(js_path, "w") as f:
            f.write(js_script)
        subprocess.run(
            ["node", js_path],
            capture_output=True, timeout=30,
            cwd="/home/ec2-user/AI"
        )
        return summary_path if os.path.exists(summary_path) else None
    except Exception as e:
        logger.warning(f"Summary screenshot failed: {e}")
        return None
    finally:
        for p in [html_path, js_path]:
            if os.path.exists(p):
                os.remove(p)


def _update_excel_with_results(plan_path: str, pw_results: List[Dict]) -> None:
    try:
        import openpyxl
        from openpyxl.drawing.image import Image as XLImage
        from openpyxl.styles import Font, Alignment, PatternFill
        from datetime import date

        wb = openpyxl.load_workbook(plan_path)
        ws = wb.active
        today = date.today().strftime("%Y-%m-%d")
        center = Alignment(horizontal="center", vertical="center", wrap_text=True)

        # ── 버그1 수정: max_row 체크 제거, 필요하면 행 자체를 추가 ──
        for i, res in enumerate(pw_results):
            row = 6 + i

            status = res.get("status", "FAIL")
            result_text = "정상" if status == "SUCCESS" else "결함"
            fill_color = "C6EFCE" if status == "SUCCESS" else "FFC7CE"
            font_color = "276221" if status == "SUCCESS" else "9C0006"

            cell_l = ws.cell(row=row, column=12, value=result_text)
            cell_l.font = Font(name="Arial", size=10, bold=True, color=font_color)
            cell_l.fill = PatternFill(start_color=fill_color, end_color=fill_color, fill_type="solid")
            cell_l.alignment = center

            ws.cell(row=row, column=13, value=today).alignment = center
            ws.cell(row=row, column=14, value="user").alignment = center

            # ── 버그2 수정: 스크린샷 생성 후 S3 URL 대신 로컬 파일로 삽입 ──
            screenshot_path = _generate_result_screenshot(
                case_idx=i + 1,
                all_results=pw_results
            )
            if screenshot_path and os.path.exists(screenshot_path):
                try:
                    img = XLImage(screenshot_path)
                    img.width = 160
                    img.height = 100
                    # 병합 셀이 있으면 삽입 위치 조정
                    ws.row_dimensions[row].height = 80
                    ws.add_image(img, f"K{row}")
                except Exception as e:
                    logger.warning(f"Screenshot insert failed row {row}: {e}")
                    ws.cell(row=row, column=11, value="-").alignment = center
            else:
                ws.cell(row=row, column=11, value="-").alignment = center

        wb.save(plan_path)
        logger.info(f"Excel updated: {plan_path} ({len(pw_results)} rows)")
    except Exception as e:
        logger.error(f"Failed to update excel: {e}")


# ── 앱 이벤트 ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    try:
        validate_settings()
        logger.info(f"Initializing DSPy with Bedrock in region: {settings.aws_region}")
        configure_bedrock_dspy(region=settings.aws_region)
        logger.info("DSPy initialization complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


# ── 백그라운드 작업 ───────────────────────────────────────────────────────────

async def generate_plan_background(
    execution_id: int, repo_url: str, branch: str,
    requirement: str, callback_url: str, auth_token: Optional[str] = None,
):
    repo_path = None
    started_at = time.time()
    try:
        logger.info(f"[{execution_id}] [PLAN] Cloning repository...")
        repo_path, file_tree, chunked_docs = await _clone_and_chunk(repo_url, branch, auth_token, execution_id)

        logger.info(f"[{execution_id}] [PLAN] Generating test plan...")
        result = await asyncio.to_thread(_run_plan_pipeline, chunked_docs, file_tree, requirement, execution_id)

        if not (isinstance(result, dict) and result.get("status") == "success"):
            raise ValueError(result.get("message", "Plan generation failed"))

        plan_path = result.get("saved_plan", "")
        test_cases = result.get("test_cases", [])

        # S3 업로드
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_s3_url = None
        if plan_path and os.path.exists(plan_path):
            s3_key = f"executions/{execution_id}/plan.xlsx"
            report_s3_url = await asyncio.to_thread(_upload_to_s3, plan_path, s3_key)

        case_results = [
            TestCaseResult(case_name=tc.get("case_name", f"Case {i+1}"), status="SUCCESS")
            for i, tc in enumerate(test_cases)
        ] or [TestCaseResult(case_name="Plan Generated", status="SUCCESS")]

        execution_store[execution_id] = {
            "execution_id": execution_id,
            "status": "COMPLETED",
            "plan_path": plan_path,
            "plan_s3_url": report_s3_url,
            "test_cases": test_cases,
        }

        duration_ms = int((time.time() - started_at) * 1000)
        await send_callback(callback_url, CallbackPayload(
            execution_id=execution_id, status="COMPLETED",
            duration_ms=duration_ms, report_s3_url=report_s3_url,
            results=case_results,
        ))

    except Exception as e:
        logger.error(f"[{execution_id}] [PLAN] Job failed: {e}")
        await send_callback(callback_url, CallbackPayload(
            execution_id=execution_id, status="FAILED",
            duration_ms=int((time.time() - started_at) * 1000),
            results=[TestCaseResult(status="FAIL", error_log=str(e))],
        ))
    finally:
        if repo_path and os.path.exists(repo_path):
            shutil.rmtree(repo_path)


async def generate_test_background(
    execution_id: int, repo_url: str, branch: str,
    requirement: str, callback_url: str, auth_token: Optional[str] = None,
):
    repo_path = None
    started_at = time.time()
    try:
        logger.info(f"[{execution_id}] [TEST] Cloning repository...")
        repo_path, file_tree, chunked_docs = await _clone_and_chunk(repo_url, branch, auth_token, execution_id)

        logger.info(f"[{execution_id}] [TEST] Generating test code...")
        result = await asyncio.to_thread(_run_test_pipeline, chunked_docs, file_tree, requirement, execution_id)

        if not (isinstance(result, dict) and result.get("status") == "success"):
            raise ValueError(result.get("message", "Test generation failed"))

        spec_path = result.get("saved_file", "")
        plan_path = result.get("saved_plan", "")

        # Playwright 실행
        logger.info(f"[{execution_id}] [TEST] Running Playwright...")
        pw_results = await asyncio.to_thread(_run_playwright, spec_path, execution_id)
        logger.info(f"[{execution_id}] [TEST] Playwright done: {len(pw_results)} results")

        # 엑셀 결과 업데이트 (각 케이스별 개별 스크린샷 → K열 삽입)
        if plan_path and os.path.exists(plan_path):
            await asyncio.to_thread(_update_excel_with_results, plan_path, pw_results)

        # 종합 스크린샷 1개 생성 → S3 업로드
        summary_s3_url = None
        summary_screenshot_path = await asyncio.to_thread(
            _generate_summary_screenshot, pw_results
        )
        if summary_screenshot_path and os.path.exists(summary_screenshot_path):
            s3_key = f"executions/{execution_id}/screenshots/summary.png"
            with open(summary_screenshot_path, "rb") as f:
                summary_bytes = f.read()
            summary_s3_url = await asyncio.to_thread(
                _upload_bytes_to_s3, summary_bytes, s3_key, "image/png"
            )

        # 엑셀, 테스트코드 S3 업로드
        report_s3_url = None
        if plan_path and os.path.exists(plan_path):
            s3_key = f"executions/{execution_id}/plan_result.xlsx"
            report_s3_url = await asyncio.to_thread(_upload_to_s3, plan_path, s3_key)

        if spec_path and os.path.exists(spec_path):
            s3_key = f"executions/{execution_id}/test.spec.js"
            await asyncio.to_thread(_upload_to_s3, spec_path, s3_key)

        # case_results 구성 (모든 케이스에 종합 스크린샷 URL 첨부)
        case_results = []
        for r in pw_results:
            case_results.append(TestCaseResult(
                case_name=r.get("case_name"),
                status=r.get("status", "FAIL"),
                error_log=r.get("error_log"),
                screenshot_s3_url=summary_s3_url,
            ))

        overall_status = "COMPLETED"
        if any(r.status == "FAIL" for r in case_results):
            logger.warning(f"[{execution_id}] Some tests failed.")

        execution_store[execution_id] = {
            "execution_id": execution_id,
            "status": overall_status,
            "test_code": result.get("test_code", ""),
            "plan_s3_url": report_s3_url,
            "results": [r.model_dump() for r in case_results],
        }

        duration_ms = int((time.time() - started_at) * 1000)
        await send_callback(callback_url, CallbackPayload(
            execution_id=execution_id, status=overall_status,
            duration_ms=duration_ms, report_s3_url=report_s3_url,
            results=case_results,
        ))

    except Exception as e:
        logger.error(f"[{execution_id}] [TEST] Job failed: {e}")
        await send_callback(callback_url, CallbackPayload(
            execution_id=execution_id, status="FAILED",
            duration_ms=int((time.time() - started_at) * 1000),
            results=[TestCaseResult(status="FAIL", error_log=str(e))],
        ))
    finally:
        if repo_path and os.path.exists(repo_path):
            shutil.rmtree(repo_path)


# ── 엔드포인트 ───────────────────────────────────────────────────────────────

@app.post(
    "/api/generate-plan",
    status_code=202,
    summary="테스트 계획서 생성",
    description=(
        "GitHub 레포지토리를 분석하여 SIT 시나리오 형식의 테스트 계획서(Excel)를 생성합니다. "
        "executionId에는 본인이 사용할 값을 넣어주시면 됩니다. 동일한 executionId로 generate-test를 호출하여 계획서 기반 테스트 코드 생성 및 실행이 가능합니다. "
        "완료 시 callback_url로 결과를 전송합니다."
    ),
    responses={422: {"model": None}},
)
async def generate_plan(request: GeneratePlanRequest, background_tasks: BackgroundTasks) -> Response:
    background_tasks.add_task(
        generate_plan_background,
        request.executionId,
        str(request.repository_url),
        request.targetBranch,
        request.requirement,
        request.callback_url,
        request.auth_token,
    )
    return Response(status_code=202)


@app.post(
    "/api/generate-test",
    status_code=202,
    summary="테스트 코드 생성 및 실행",
    description=(
        "generate-plan으로 생성된 계획서를 기반으로 Playwright 테스트 코드를 생성하고 실행합니다. "
        "실행 결과가 계획서 Excel에 반영되며, 완료 시 callback_url로 결과를 전송합니다. "
        "반드시 generate-plan을 먼저 실행해야 합니다."
    ),
    responses={422: {"model": None}},
)
async def generate_test(request: GenerateTestRequest, background_tasks: BackgroundTasks) -> Response:
    background_tasks.add_task(
        generate_test_background,
        request.executionId,
        str(request.repository_url),
        request.targetBranch,
        request.requirement,
        request.callback_url,
        request.auth_token,
    )
    return Response(status_code=202)


@app.get(
    "/api/result/{execution_id}",
    summary="생성 결과 조회",
    description="테스트 계획서 S3 URL과 생성된 Playwright 테스트 코드를 조회합니다.",
)
async def get_result(execution_id: int):
    result = execution_store.get(execution_id)
    if not result:
        raise HTTPException(status_code=404, detail="Result not found. Job may still be running.")
    return result


@app.get(
    "/api/health",
    summary="서버 상태 확인",
    responses={200: {"content": {"application/json": {"example": {"status": "healthy"}}}}},
)
async def health_check():
    return {
        "status": "healthy",
        "region": settings.aws_region,
        "model": settings.bedrock_model,
        "environment": settings.environment,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=(settings.environment == "development"),
    )