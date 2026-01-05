"""
FastAPI 메인 애플리케이션

Playwright 테스트 코드 생성 API를 제공합니다.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

# 프로젝트 모듈
from config import settings, validate_settings, print_settings
from src.dspy_modules import PlaywrightTestGenerator, configure_dspy
from src.embeddings import get_embedding_generator
from src.database import DatabaseManager, Repository, TestScript
from src.langchain_integration import RAGPipeline
from src.storage import FileManager
from src.gpt import GPTClient
from src.claude import ClaudeClient

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="PROBE - Playwright Test Generator",
    description="GitHub 코드를 분석하여 Playwright 테스트 코드를 자동 생성합니다",
    version="1.0.0"
)

# CORS 설정 (프론트엔드 연동용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 컴포넌트
db_manager: Optional[DatabaseManager] = None
rag_pipeline: Optional[RAGPipeline] = None
file_manager: Optional[FileManager] = None
test_generator: Optional[PlaywrightTestGenerator] = None
gpt_client: Optional[GPTClient] = None
claude_client: Optional[ClaudeClient] = None


# ===== Pydantic 모델 (요청/응답) =====

class TestGenerationRequest(BaseModel):
    """테스트 생성 요청"""
    repository_url: str = Field(..., description="GitHub 저장소 URL")
    requirement: str = Field(..., description="테스트 요구사항")
    test_type: str = Field(default="e2e", description="테스트 타입 (e2e, unit, integration)")
    base_url: str = Field(..., description="테스트 대상 애플리케이션 URL")
    
    class Config:
        schema_extra = {
            "example": {
                "repository_url": "https://github.com/user/ecommerce-app",
                "requirement": "사용자 로그인 기능 E2E 테스트",
                "test_type": "e2e",
                "base_url": "https://shop.example.com"
            }
        }


class TestScriptInfo(BaseModel):
    """테스트 스크립트 정보"""
    filename: str
    url: str
    download_url: str
    content_preview: str
    full_content_lines: int


class TestMetadata(BaseModel):
    """테스트 메타데이터"""
    lines_of_code: int
    test_cases: List[str]
    test_description: str
    framework: str = "playwright"
    language: str = "javascript"


class RepositoryInfo(BaseModel):
    """저장소 정보"""
    id: int
    name: str
    url: str


class TestGenerationResponse(BaseModel):
    """테스트 생성 응답"""
    test_script: TestScriptInfo
    metadata: TestMetadata
    repository: RepositoryInfo


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    models: Dict[str, Any]
    database: Dict[str, Any]
    timestamp: str


# ===== 초기화 =====

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 초기화"""
    global db_manager, rag_pipeline, file_manager, test_generator
    global gpt_client, claude_client
    
    logger.info("=" * 60)
    logger.info("Starting PROBE - Playwright Test Generator")
    logger.info("=" * 60)
    
    # 설정 검증
    print_settings()
    validate_settings()
    
    try:
        # 1. Database 초기화
        logger.info("Initializing DatabaseManager...")
        db_manager = DatabaseManager(settings.database_url)
        db_manager.create_tables()
        logger.info("✅ DatabaseManager initialized")
        
        # 2. RAG Pipeline 초기화
        logger.info("Initializing RAG Pipeline...")
        rag_pipeline = RAGPipeline(
            collection_name="code_embeddings",
            embedding_model=settings.embedding_model
        )
        logger.info("✅ RAG Pipeline initialized")
        
        # 3. FileManager 초기화 (S3/MinIO)
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            logger.info("Initializing FileManager...")
            file_manager = FileManager(
                endpoint_url=settings.s3_endpoint_url,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region,
                bucket_name=settings.s3_bucket_name
            )
            logger.info("✅ FileManager initialized")
        else:
            logger.warning("⚠️ FileManager not initialized (no S3 credentials)")
        
        # 4. AI 모델 초기화
        logger.info("Initializing AI models...")
        
        # GPT 클라이언트
        if settings.is_gpt_enabled:
            gpt_client = GPTClient(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model
            )
            logger.info("✅ GPT Client initialized")
        
        # Claude 클라이언트
        if settings.is_claude_enabled:
            claude_client = ClaudeClient(
                api_key=settings.anthropic_api_key,
                model=settings.claude_model
            )
            logger.info("✅ Claude Client initialized")
        
        # 5. DSPy 설정 (기본 모델: GPT)
        if settings.is_gpt_enabled:
            configure_dspy(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model
            )
            logger.info("✅ DSPy configured with GPT")
        
        # 6. PlaywrightTestGenerator 초기화
        test_generator = PlaywrightTestGenerator()
        logger.info("✅ PlaywrightTestGenerator initialized")
        
        logger.info("=" * 60)
        logger.info("✅ Server startup complete!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 정리"""
    logger.info("Shutting down server...")
    
    if db_manager:
        db_manager.close()
        logger.info("✅ Database connections closed")


# ===== API 엔드포인트 =====

@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "PROBE - Playwright Test Generator API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    서버 헬스체크
    
    Returns:
        HealthResponse: 서버 상태 정보
    """
    try:
        # 모델 상태
        models_status = {
            "gpt": {
                "enabled": settings.is_gpt_enabled,
                "status": "connected" if gpt_client else "not_initialized"
            },
            "claude": {
                "enabled": settings.is_claude_enabled,
                "status": "connected" if claude_client else "not_initialized"
            }
        }
        
        # 데이터베이스 상태
        db_status = {}
        if db_manager:
            stats = db_manager.get_stats()
            db_status = {
                "status": "connected",
                "repositories": stats.get("repositories", 0),
                "code_chunks": stats.get("code_chunks", 0),
                "test_scripts": stats.get("test_scripts", 0)
            }
        else:
            db_status = {"status": "not_initialized"}
        
        return {
            "status": "healthy",
            "models": models_status,
            "database": db_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/api/generate-test",
    response_model=TestGenerationResponse,
    tags=["Test Generation"]
)
async def generate_playwright_test(request: TestGenerationRequest):
    """
    Playwright 테스트 코드 생성
    
    **워크플로우:**
    1. GitHub에서 코드 다운로드
    2. 코드 청킹 및 벡터화
    3. RAG로 관련 코드 검색
    4. AI로 Playwright 테스트 생성
    5. S3/MinIO에 저장
    6. DB에 메타데이터 저장
    
    Args:
        request: 테스트 생성 요청
    
    Returns:
        TestGenerationResponse: 생성된 테스트 정보
    
    Raises:
        HTTPException: 생성 실패 시
    """
    try:
        logger.info(f"Generating test for: {request.repository_url}")
        
        # 1. 저장소 처리 (이미 있으면 조회, 없으면 생성)
        repo = db_manager.get_repository_by_url(request.repository_url)
        if not repo:
            # GitHub에서 정보 추출 (간단한 파싱)
            repo_name = request.repository_url.rstrip('/').split('/')[-1]
            repo_owner = request.repository_url.rstrip('/').split('/')[-2]
            
            repo = db_manager.create_repository(
                url=request.repository_url,
                name=repo_name,
                owner=repo_owner
            )
            logger.info(f"Created repository: {repo.name} (id={repo.id})")
        else:
            logger.info(f"Found existing repository: {repo.name} (id={repo.id})")
        
        # 2. RAG 검색 (관련 코드 찾기)
        logger.info("Searching for relevant code...")
        context = rag_pipeline.retrieve_context(
            requirement=request.requirement,
            top_k=5,
            max_chars=4000
        )
        
        if not context:
            logger.warning("No code context found. Using empty context.")
            context = "No relevant code found in the repository."
        
        logger.info(f"Retrieved context: {len(context)} chars")
        
        # 3. AI 테스트 생성
        logger.info("Generating Playwright test code...")
        result = test_generator.generate_to_dict(
            requirement=request.requirement,
            code_context=context,
            base_url=request.base_url
        )
        
        logger.info(
            f"Test generated: {result['lines_of_code']} lines, "
            f"{len(result['test_cases'])} test cases"
        )
        
        # 4. 파일 저장 (S3/MinIO)
        file_info = None
        if file_manager:
            logger.info("Saving test script to S3/MinIO...")
            
            # 테스트 이름 추출 (요구사항에서 첫 단어)
            test_name = request.requirement.split()[0].lower()
            
            file_info = file_manager.save_test_script(
                test_code=result["test_code"],
                repository_name=repo.name,
                test_name=test_name,
                metadata={
                    "requirement": request.requirement,
                    "test_type": request.test_type,
                    "base_url": request.base_url
                }
            )
            logger.info(f"Test script saved: {file_info['filename']}")
        else:
            logger.warning("FileManager not available. Test script not saved to storage.")
            file_info = {
                "filename": f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.spec.js",
                "storage_url": "local://not_saved",
                "download_url": "local://not_saved",
                "s3_key": "local://not_saved"
            }
        
        # 5. DB에 메타데이터 저장
        logger.info("Saving metadata to database...")
        test_script = db_manager.create_test_script(
            repository_id=repo.id,
            requirement=request.requirement,
            test_code=result["test_code"],
            filename=file_info["filename"],
            test_type=request.test_type,
            base_url=request.base_url,
            test_description=result["test_description"],
            test_cases=result["test_cases"],
            lines_of_code=result["lines_of_code"],
            storage_url=file_info["storage_url"],
            download_url=file_info["download_url"],
            ai_model=settings.ai_model_mode,
            code_context=context[:500],  # 처음 500자만 저장
            metadata={
                "processing_time": "calculated_later",
                "chunks_used": 5
            }
        )
        logger.info(f"TestScript saved to DB: id={test_script.id}")
        
        # 6. 응답 생성
        return TestGenerationResponse(
            test_script=TestScriptInfo(
                filename=file_info["filename"],
                url=file_info["storage_url"],
                download_url=file_info["download_url"],
                content_preview=result["test_code"][:300] + "...",
                full_content_lines=result["lines_of_code"]
            ),
            metadata=TestMetadata(
                lines_of_code=result["lines_of_code"],
                test_cases=result["test_cases"],
                test_description=result["test_description"],
                framework="playwright",
                language="javascript"
            ),
            repository=RepositoryInfo(
                id=repo.id,
                name=repo.name,
                url=repo.url
            )
        )
    
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test generation failed: {str(e)}"
        )


@app.get("/api/tests/{script_id}", tags=["Test Scripts"])
async def get_test_script(script_id: int):
    """
    테스트 스크립트 조회
    
    Args:
        script_id: 테스트 스크립트 ID
    
    Returns:
        dict: 테스트 스크립트 정보
    """
    try:
        script = db_manager.get_test_script_by_id(script_id)
        
        if not script:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Test script with id={script_id} not found"
            )
        
        return script.to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get test script: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/repositories/{repo_id}/tests", tags=["Test Scripts"])
async def get_repository_tests(
    repo_id: int,
    test_type: Optional[str] = None,
    limit: int = 100
):
    """
    특정 저장소의 테스트 스크립트 목록 조회
    
    Args:
        repo_id: 저장소 ID
        test_type: 테스트 타입 필터 (선택)
        limit: 최대 개수
    
    Returns:
        list: 테스트 스크립트 목록
    """
    try:
        scripts = db_manager.get_test_scripts_by_repository(
            repository_id=repo_id,
            test_type=test_type,
            limit=limit
        )
        
        return [script.to_dict() for script in scripts]
    
    except Exception as e:
        logger.error(f"Failed to get repository tests: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/stats", tags=["Statistics"])
async def get_statistics():
    """
    시스템 통계 조회
    
    Returns:
        dict: 전체 통계 정보
    """
    try:
        stats = db_manager.get_stats()
        
        return {
            "database": stats,
            "storage": {
                "enabled": file_manager is not None,
                "bucket": settings.s3_bucket_name if file_manager else None
            },
            "models": {
                "gpt_enabled": settings.is_gpt_enabled,
                "claude_enabled": settings.is_claude_enabled,
                "mode": settings.ai_model_mode
            }
        }
    
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# 개발용 엔드포인트 (프로덕션에서는 제거)
@app.get("/api/debug/config", tags=["Debug"])
async def get_config():
    """현재 설정 확인 (개발용)"""
    return {
        "ai_model_mode": settings.ai_model_mode,
        "gpt_enabled": settings.is_gpt_enabled,
        "claude_enabled": settings.is_claude_enabled,
        "database_url": settings.database_url.split('@')[0] + "@***",  # 비밀번호 숨김
        "s3_configured": file_manager is not None
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=True,
        log_level=settings.log_level.lower()
    )