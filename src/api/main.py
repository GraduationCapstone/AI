"""
FastAPI 서버

Spring Boot와 통신하는 AI 서버입니다.
GitHub 저장소를 분석하여 Playwright 테스트 코드를 생성합니다.

파이프라인:
  STEP 1 — 레포 분석 → 테스트 계획서 생성
  STEP 2 — 테스트 계획서 → Playwright 테스트 코드 생성
"""

from fastapi import FastAPI, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
import asyncio
import logging
import tempfile
import shutil
import subprocess
import os
import time
import httpx

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 모델 ─────────────────────────────────────────────────────────────────────

class GenerateTestRequest(BaseModel):
    execution_id: int
    repository_url: HttpUrl
    branch: str = "main"
    requirement: str          # Spring이 채운 시나리오 프롬프트 전문
    base_url: str
    auth_token: Optional[str] = None
    callback_url: str


class TestCaseResult(BaseModel):
    case_name: Optional[str] = None
    status: str                       # "SUCCESS" | "FAIL"
    error_log: Optional[str] = None
    screenshot_s3_url: Optional[str] = None


class CallbackPayload(BaseModel):
    execution_id: int
    status: str                       # "COMPLETED" | "FAILED"
    duration_ms: Optional[int] = None
    report_s3_url: Optional[str] = None
    results: Optional[List[TestCaseResult]] = None


# ── 블로킹 함수 (asyncio.to_thread 전용) ─────────────────────────────────────

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
        dirs[:] = [d for d in dirs if d != ".git"]
        level = root.replace(path, "").count(os.sep)
        tree.append(f"{'    ' * level}{os.path.basename(root)}/")
        for f in files:
            tree.append(f"{'    ' * (level + 1)}{f}")
    return "\n".join(tree)


def _load_documents(repo_path: str) -> List[Document]:
    documents = []
    target_extensions = (".js", ".ts", ".tsx", ".java", ".html", ".css")
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(target_extensions):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    rel_path = os.path.relpath(file_path, repo_path)
                    documents.append(Document(page_content=content, metadata={"source": rel_path}))
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")
    return documents


def _run_rag_pipeline(
    chunked_docs,
    file_tree: str,
    requirement: str,
    base_url: str,
    execution_id: int,
) -> dict:
    """STEP 1 (계획서) + STEP 2 (코드) 순차 실행"""
    generator = RAGPlaywrightGenerator(region=settings.aws_region)
    generator.index_documents(chunked_docs, file_tree=file_tree)
    return generator.generate_test(
        requirement=requirement,
        base_url=base_url,
        top_k=settings.rag_top_k,
        execution_id=execution_id,
    )


def _validate_syntax(test_code: str) -> Optional[str]:
    """Node.js 구문 검증. 오류 없으면 None 반환"""
    with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=True) as tmp:
        tmp.write(test_code)
        tmp.flush()
        try:
            subprocess.run(["node", "-c", tmp.name], check=True, capture_output=True, text=True)
            return None
        except subprocess.CalledProcessError as e:
            lines = e.stderr.splitlines()
            return lines[-1] if lines else str(e)


# ── 콜백 ─────────────────────────────────────────────────────────────────────

async def send_callback(callback_url: str, payload: CallbackPayload) -> None:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                callback_url,
                json=payload.model_dump(),
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            logger.info(
                f"Callback OK → {callback_url} "
                f"(execution_id={payload.execution_id}, status={payload.status})"
            )
    except Exception as e:
        logger.error(
            f"Callback FAILED → {callback_url} "
            f"(execution_id={payload.execution_id}): {e}"
        )


# ── 백그라운드 작업 ───────────────────────────────────────────────────────────

async def generate_test_background(
    execution_id: int,
    repo_url: str,
    branch: str,
    requirement: str,
    base_url: str,
    callback_url: str,
    auth_token: Optional[str] = None,
):
    repo_path = None
    started_at = time.time()

    try:
        # 1. 클론
        logger.info(f"[{execution_id}] Cloning repository...")
        repo_path = await asyncio.to_thread(_clone_sync, repo_url, branch, auth_token)

        # 2. 파일 트리 + 문서 로드 (병렬)
        logger.info(f"[{execution_id}] Scanning & loading files...")
        file_tree, documents = await asyncio.gather(
            asyncio.to_thread(_get_file_tree, repo_path),
            asyncio.to_thread(_load_documents, repo_path),
        )

        if not documents:
            raise ValueError("분석할 소스 코드 파일(.js, .html 등)을 찾을 수 없습니다.")

        # 3. 청킹
        logger.info(f"[{execution_id}] Chunking documents...")
        chunker = CodeChunker(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap
        )
        chunked_docs = await asyncio.to_thread(chunker.split_by_file_extension, documents)

        # 4. STEP 1 (계획서) + STEP 2 (코드) 순차 실행
        logger.info(f"[{execution_id}] Running 2-step RAG pipeline...")
        result = await asyncio.to_thread(
            _run_rag_pipeline, chunked_docs, file_tree, requirement, base_url, execution_id
        )

        if not (isinstance(result, dict) and result.get("status") == "success"):
            raise ValueError(result.get("message", "Unknown generation error"))

        test_code = result.get("test_code", "")
        test_cases = result.get("test_cases", [])  # 테스트 계획서 케이스 목록

        # 5. 문법 검증
        logger.info(f"[{execution_id}] Validating syntax...")
        validation_error = await asyncio.to_thread(_validate_syntax, test_code)

        # 6. 콜백용 results 구성 — 테스트 계획서의 각 케이스 반영
        if test_cases:
            case_results = [
                TestCaseResult(
                    case_name=tc.get("case_name", tc.get("name", f"Case {i+1}")),
                    status="FAIL" if validation_error else "SUCCESS",
                    error_log=validation_error if validation_error else None,
                )
                for i, tc in enumerate(test_cases)
            ]
        else:
            # 계획서 파싱 실패 시 단일 결과로 대체
            case_results = [TestCaseResult(
                case_name="Syntax Validation",
                status="FAIL" if validation_error else "SUCCESS",
                error_log=validation_error,
            )]

        if validation_error:
            logger.error(f"[{execution_id}] Syntax error: {validation_error}")
        else:
            logger.info(f"[{execution_id}] Syntax validation passed.")

        duration_ms = int((time.time() - started_at) * 1000)

        await send_callback(callback_url, CallbackPayload(
            execution_id=execution_id,
            status="COMPLETED",
            duration_ms=duration_ms,
            results=case_results,
        ))

    except Exception as e:
        logger.error(f"[{execution_id}] Job failed: {e}")
        await send_callback(callback_url, CallbackPayload(
            execution_id=execution_id,
            status="FAILED",
            duration_ms=int((time.time() - started_at) * 1000),
            results=[TestCaseResult(status="FAIL", error_log=str(e))],
        ))

    finally:
        if repo_path and os.path.exists(repo_path):
            shutil.rmtree(repo_path)


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


# ── 엔드포인트 ───────────────────────────────────────────────────────────────

@app.post(
    "/api/generate-test",
    status_code=202,
    summary="테스트 코드 생성 요청",
    description=(
        "GitHub 레포지토리를 분석하여 테스트 계획서 및 Playwright 테스트 코드를 생성합니다. "
        "작업은 비동기로 실행되며, 완료 시 callback_url로 결과를 전송합니다."
    ),
    responses={422: {"model": None}},
)
async def generate_test(
    request: GenerateTestRequest,
    background_tasks: BackgroundTasks,
) -> Response:
    background_tasks.add_task(
        generate_test_background,
        request.execution_id,
        str(request.repository_url),
        request.branch,
        request.requirement,
        request.base_url,
        request.callback_url,
        request.auth_token,
    )
    return Response(status_code=202)


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