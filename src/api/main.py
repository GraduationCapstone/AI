"""
FastAPI 서버

Spring Boot와 통신하는 AI 서버입니다.
GitHub 저장소를 분석하여 Playwright 테스트 코드를 생성합니다.

주요 엔드포인트:
- POST /api/generate-test: 테스트 코드 생성 (GitHub 저장소 기반)
- GET /api/health: 서버 상태 확인
- GET /api/status/{job_id}: 작업 상태 조회
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Dict, Any
from enum import Enum
import logging
import uuid
from datetime import datetime
import tempfile
import shutil
import subprocess

from src.dspy_modules import configure_bedrock_dspy, RAGPlaywrightGenerator
from src.langchain_integration import CodeChunker, RAGPipeline
from langchain_core.documents import Document

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="PROBE AI Server",
    description="AI 기반 Playwright 테스트 코드 자동 생성 서버",
    version="2.0.0"
)

# CORS 설정 (Spring Boot와 통신)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DSPy 초기화 (앱 시작 시 1회)
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 DSPy 초기화"""
    try:
        logger.info("Initializing DSPy with Bedrock...")
        configure_bedrock_dspy(region="us-east-1")
        logger.info("DSPy initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize DSPy: {e}")
        raise


# ===== 데이터 모델 =====

class JobStatus(str, Enum):
    """작업 상태"""
    PENDING = "pending"
    CLONING = "cloning"
    INDEXING = "indexing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerateTestRequest(BaseModel):
    """테스트 생성 요청"""
    repository_url: HttpUrl
    branch: Optional[str] = "main"
    requirement: str
    base_url: HttpUrl
    auth_token: Optional[str] = None  # GitHub 인증 토큰 (private repo)
    
    class Config:
        schema_extra = {
            "example": {
                "repository_url": "https://github.com/username/repo",
                "branch": "main",
                "requirement": "사용자 로그인 기능 E2E 테스트",
                "base_url": "https://example.com",
                "auth_token": "ghp_xxxxxxxxxxxx"
            }
        }


class TestGenerationResponse(BaseModel):
    """테스트 생성 응답"""
    job_id: str
    status: JobStatus
    message: str


class TestResult(BaseModel):
    """테스트 결과"""
    job_id: str
    status: JobStatus
    test_code: Optional[str] = None
    test_description: Optional[str] = None
    test_cases: Optional[List[str]] = None
    lines_of_code: Optional[int] = None
    error: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


# ===== 작업 저장소 (메모리) =====
# 프로덕션에서는 Redis 등 사용 권장
job_store: Dict[str, TestResult] = {}


# ===== 헬퍼 함수 =====

def clone_repository(repo_url: str, branch: str, auth_token: Optional[str] = None) -> str:
    """
    GitHub 저장소 클론
    
    Args:
        repo_url: 저장소 URL
        branch: 브랜치명
        auth_token: GitHub 토큰 (private repo용)
    
    Returns:
        str: 클론된 디렉토리 경로
    """
    temp_dir = tempfile.mkdtemp(prefix="probe_repo_")
    
    try:
        # 인증 토큰이 있으면 URL에 포함
        if auth_token:
            # https://github.com/user/repo -> https://token@github.com/user/repo
            clone_url = repo_url.replace("https://", f"https://{auth_token}@")
        else:
            clone_url = repo_url
        
        # git clone 실행
        logger.info(f"Cloning repository: {repo_url} (branch: {branch})")
        result = subprocess.run(
            ["git", "clone", "-b", branch, "--depth", "1", clone_url, temp_dir],
            capture_output=True,
            text=True,
            timeout=300  # 5분 타임아웃
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {result.stderr}")
        
        logger.info(f"Repository cloned to: {temp_dir}")
        return temp_dir
    
    except Exception as e:
        # 실패 시 임시 디렉토리 정리
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def get_file_tree(repo_path: str) -> str:
    """
    파일 트리 생성 (Tree-First 전략용)
    
    Args:
        repo_path: 저장소 경로
    
    Returns:
        str: 파일 트리 문자열
    """
    try:
        result = subprocess.run(
            ["tree", "-L", "3", "-I", "node_modules|__pycache__|.git", repo_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            return result.stdout
        else:
            logger.warning("tree command failed, using basic structure")
            return f"Repository: {repo_path}"
    
    except FileNotFoundError:
        logger.warning("tree command not found, skipping file tree")
        return f"Repository: {repo_path}"
    except Exception as e:
        logger.error(f"Failed to generate file tree: {e}")
        return f"Repository: {repo_path}"


def scan_code_files(repo_path: str, extensions: List[str] = [".py", ".js", ".java"]) -> List[Document]:
    """
    코드 파일 스캔 및 Document 생성
    
    Args:
        repo_path: 저장소 경로
        extensions: 스캔할 파일 확장자
    
    Returns:
        List[Document]: LangChain Document 리스트
    """
    import os
    
    documents = []
    
    for root, dirs, files in os.walk(repo_path):
        # 제외 디렉토리
        dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', 'venv']]
        
        for file in files:
            # 확장자 필터링
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)
                
                try:
                    # 파일 읽기
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Document 생성
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": relative_path,
                            "language": file.split('.')[-1]
                        }
                    )
                    documents.append(doc)
                
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
    
    logger.info(f"Scanned {len(documents)} code files")
    return documents


async def generate_test_background(
    job_id: str,
    repo_url: str,
    branch: str,
    requirement: str,
    base_url: str,
    auth_token: Optional[str] = None
):
    """
    백그라운드에서 테스트 코드 생성 (비동기)
    
    Args:
        job_id: 작업 ID
        repo_url: 저장소 URL
        branch: 브랜치명
        requirement: 테스트 요구사항
        base_url: 테스트 대상 URL
        auth_token: GitHub 토큰
    """
    repo_path = None
    
    try:
        # 1. 저장소 클론
        job_store[job_id].status = JobStatus.CLONING
        logger.info(f"[{job_id}] Cloning repository...")
        repo_path = clone_repository(repo_url, branch, auth_token)
        
        # 2. 파일 트리 생성
        file_tree = get_file_tree(repo_path)
        
        # 3. 코드 파일 스캔
        logger.info(f"[{job_id}] Scanning code files...")
        raw_documents = scan_code_files(repo_path)
        
        if not raw_documents:
            raise ValueError("No code files found in repository")
        
        # 4. 코드 청킹
        job_store[job_id].status = JobStatus.INDEXING
        logger.info(f"[{job_id}] Chunking code...")
        chunker = CodeChunker(chunk_size=1000, chunk_overlap=200)
        chunked_documents = chunker.split_by_file_extension(raw_documents)
        
        # 5. RAG Generator 초기화 및 인덱싱
        logger.info(f"[{job_id}] Indexing documents...")
        generator = RAGPlaywrightGenerator(region="us-east-1")
        generator.index_documents(chunked_documents, file_tree=file_tree)
        
        # 6. 테스트 코드 생성
        job_store[job_id].status = JobStatus.GENERATING
        logger.info(f"[{job_id}] Generating test code...")
        result = generator.generate_test(
            requirement=requirement,
            base_url=base_url,
            top_k=5
        )
        
        # 7. 결과 저장
        job_store[job_id].status = JobStatus.COMPLETED
        job_store[job_id].test_code = result["test_code"]
        job_store[job_id].test_description = result["test_description"]
        job_store[job_id].test_cases = result["test_cases"]
        job_store[job_id].lines_of_code = result["lines_of_code"]
        job_store[job_id].completed_at = datetime.utcnow().isoformat()
        
        logger.info(f"[{job_id}] Test generation completed successfully")
    
    except Exception as e:
        logger.error(f"[{job_id}] Test generation failed: {e}")
        job_store[job_id].status = JobStatus.FAILED
        job_store[job_id].error = str(e)
        job_store[job_id].completed_at = datetime.utcnow().isoformat()
    
    finally:
        # 임시 디렉토리 정리
        if repo_path:
            try:
                shutil.rmtree(repo_path, ignore_errors=True)
                logger.info(f"[{job_id}] Cleaned up repository: {repo_path}")
            except Exception as e:
                logger.warning(f"[{job_id}] Failed to cleanup: {e}")


# ===== API 엔드포인트 =====

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "PROBE AI Server",
        "version": "2.0.0",
        "status": "running"
    }


@app.get("/api/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_jobs": len([j for j in job_store.values() if j.status in [JobStatus.PENDING, JobStatus.CLONING, JobStatus.INDEXING, JobStatus.GENERATING]])
    }


@app.post("/api/generate-test", response_model=TestGenerationResponse)
async def generate_test(
    request: GenerateTestRequest,
    background_tasks: BackgroundTasks
):
    """
    테스트 코드 생성 요청
    
    GitHub 저장소를 클론하여 코드를 분석하고,
    Playwright 테스트 코드를 생성합니다.
    
    Args:
        request: 테스트 생성 요청
        background_tasks: FastAPI 백그라운드 작업
    
    Returns:
        TestGenerationResponse: 작업 ID 및 상태
    """
    try:
        # 작업 ID 생성
        job_id = str(uuid.uuid4())
        
        # 작업 저장소에 등록
        job_store[job_id] = TestResult(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow().isoformat()
        )
        
        # 백그라운드 작업 시작
        background_tasks.add_task(
            generate_test_background,
            job_id=job_id,
            repo_url=str(request.repository_url),
            branch=request.branch,
            requirement=request.requirement,
            base_url=str(request.base_url),
            auth_token=request.auth_token
        )
        
        logger.info(f"Test generation job created: {job_id}")
        
        return TestGenerationResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Test generation started. Use /api/status/{job_id} to check progress."
        )
    
    except Exception as e:
        logger.error(f"Failed to create test generation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status/{job_id}", response_model=TestResult)
async def get_job_status(job_id: str):
    """
    작업 상태 조회
    
    Args:
        job_id: 작업 ID
    
    Returns:
        TestResult: 작업 상태 및 결과
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    return job_store[job_id]


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """
    작업 삭제
    
    Args:
        job_id: 작업 ID
    
    Returns:
        dict: 삭제 확인
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    del job_store[job_id]
    logger.info(f"Job deleted: {job_id}")
    
    return {"message": f"Job {job_id} deleted successfully"}


# ===== 개발용 엔드포인트 =====

@app.get("/api/jobs")
async def list_jobs():
    """모든 작업 조회 (개발용)"""
    return {
        "total": len(job_store),
        "jobs": [
            {
                "job_id": job.job_id,
                "status": job.status,
                "created_at": job.created_at
            }
            for job in job_store.values()
        ]
    }


# 사용 예시
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드
        log_level="info"
    )