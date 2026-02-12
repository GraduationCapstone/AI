"""
FastAPI 서버

Spring Boot와 통신하는 AI 서버입니다.
GitHub 저장소를 분석하여 Playwright 테스트 코드를 생성합니다.
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
import os

# 설정 및 AI 모듈 임포트
from config.config import settings, validate_settings
from src.dspy_modules import configure_bedrock_dspy, RAGPlaywrightGenerator
from src.langchain_integration import CodeChunker, RAGPipeline
from langchain_core.documents import Document

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="PROBE AI Server",
    description="AI 기반 Playwright 테스트 코드 자동 생성 서버",
    version="2.0.0"
)

# CORS 설정 (Spring Boot 및 프론트엔드 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 인메모리 작업 저장소
job_store: Dict[str, 'TestResult'] = {}

class JobStatus(str, Enum):
    PENDING = "pending"
    CLONING = "cloning"
    SCANNING = "scanning"
    INDEXING = "indexing"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"

class GenerateTestRequest(BaseModel):
    repository_url: HttpUrl
    branch: str = "main"
    requirement: str
    base_url: str
    auth_token: Optional[str] = None

class TestResult(BaseModel):
    job_id: str
    status: JobStatus
    test_code: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

# DSPy 및 설정 초기화 (앱 시작 시 실행)
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 설정 검증 및 DSPy 초기화"""
    try:
        # 1. 환경 변수 및 설정 검증 (IAM Role 사용 확인 포함)
        validate_settings()
        
        # 2. 리전 하드코딩 수정: settings.aws_region 사용
        logger.info(f"Initializing DSPy with Bedrock in region: {settings.aws_region}")
        configure_bedrock_dspy(region=settings.aws_region)
        
        logger.info("DSPy initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize server settings: {e}")
        raise

def clone_repository(repo_url: str, branch: str, token: Optional[str] = None) -> str:
    """GitHub 저장소를 임시 디렉토리에 클론"""
    tmp_dir = tempfile.mkdtemp()
    
    # 토큰이 있는 경우 URL에 삽입
    url = str(repo_url)
    if token:
        url = url.replace("https://", f"https://{token}@")
    
    try:
        subprocess.run(
            ["git", "clone", "-b", branch, "--depth", "1", url, tmp_dir],
            check=True,
            capture_output=True
        )
        return tmp_dir
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp_dir)
        raise Exception(f"Git clone failed: {e.stderr.decode()}")

def get_file_tree(path: str) -> str:
    """파일 트리 구조 생성 (분석용)"""
    tree = []
    for root, dirs, files in os.walk(path):
        if '.git' in dirs:
            dirs.remove('.git')
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        tree.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            tree.append(f"{sub_indent}{f}")
    return "\n".join(tree)

async def generate_test_background(
    job_id: str,
    repo_url: str,
    branch: str,
    requirement: str,
    base_url: str,
    auth_token: Optional[str] = None
):
    """백그라운드에서 테스트 코드 생성 로직 수행"""
    repo_path = None
    try:
        # 1. 클론
        job_store[job_id].status = JobStatus.CLONING
        job_store[job_id].updated_at = datetime.now()
        repo_path = clone_repository(repo_url, branch, auth_token)
        
        # 2. 파일 스캔 및 분석
        job_store[job_id].status = JobStatus.SCANNING
        file_tree = get_file_tree(repo_path)
        
        # 3. 코드 로딩 및 청킹
        job_store[job_id].status = JobStatus.INDEXING
        chunker = CodeChunker(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap
        )
        # 소스 파일 로드 로직 (예시: .js, .ts, .java 등)
        documents = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith(('.js', '.ts', '.tsx', '.java', '.html')):
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append(Document(page_content=content, metadata={"source": file}))
        
        chunked_docs = chunker.split_documents(documents)
        
        # 4. RAG Generator 초기화 및 실행 (리전 설정 반영)
        generator = RAGPlaywrightGenerator(region=settings.aws_region)
        generator.index_documents(chunked_docs, file_tree=file_tree)
        
        job_store[job_id].status = JobStatus.GENERATING
        test_code = generator.generate_test(
            requirement=requirement,
            base_url=base_url,
            top_k=settings.rag_top_k
        )
        
        # 5. 완료 처리
        job_store[job_id].test_code = test_code
        job_store[job_id].status = JobStatus.COMPLETED
        job_store[job_id].updated_at = datetime.now()
        
    except Exception as e:
        logger.error(f"Error in background job {job_id}: {str(e)}")
        job_store[job_id].status = JobStatus.FAILED
        job_store[job_id].error = str(e)
        job_store[job_id].updated_at = datetime.now()
    finally:
        if repo_path and os.path.exists(repo_path):
            shutil.rmtree(repo_path)

@app.post("/api/generate-test", response_model=Dict[str, str])
async def generate_test(request: GenerateTestRequest, background_tasks: BackgroundTasks):
    """테스트 코드 생성 요청 엔드포인트"""
    job_id = str(uuid.uuid4())
    now = datetime.now()
    
    job_store[job_id] = TestResult(
        job_id=job_id,
        status=JobStatus.PENDING,
        created_at=now,
        updated_at=now
    )
    
    background_tasks.add_task(
        generate_test_background,
        job_id,
        str(request.repository_url),
        request.branch,
        request.requirement,
        request.base_url,
        request.auth_token
    )
    
    return {"job_id": job_id}

@app.get("/api/status/{job_id}", response_model=TestResult)
async def get_status(job_id: str):
    """작업 상태 및 결과 조회"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_store[job_id]

@app.get("/api/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "region": settings.aws_region,
        "model": settings.bedrock_model,
        "environment": settings.environment
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=(settings.environment == "development")
    )