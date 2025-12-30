"""
데이터베이스 모듈

이 모듈은 PostgreSQL + pgvector를 사용한 데이터베이스 관리를 제공합니다.

주요 컴포넌트:
- Base: SQLAlchemy 선언적 베이스
- Repository: GitHub 저장소 모델
- CodeChunk: 코드 청크 + 벡터 임베딩 모델
- AnalysisResult: AI 분석 결과 모델
- DatabaseManager: 데이터베이스 CRUD 작업 관리자

사용 예시:
    >>> from config import settings
    >>> from src.database import DatabaseManager, Repository, CodeChunk
    >>> 
    >>> # 1. DatabaseManager 생성
    >>> db_manager = DatabaseManager(settings.database_url)
    >>> 
    >>> # 2. 테이블 생성
    >>> db_manager.create_tables()
    >>> 
    >>> # 3. 저장소 생성
    >>> repo = db_manager.create_repository(
    ...     url="https://github.com/user/repo",
    ...     name="repo",
    ...     owner="user"
    ... )
    >>> 
    >>> # 4. 코드 청크 생성
    >>> chunk = db_manager.create_code_chunk(
    ...     repository_id=repo.id,
    ...     file_path="auth.py",
    ...     content="def login(): pass",
    ...     embedding=[0.1, 0.2, ...],  # 384차원
    ...     language="python"
    ... )
    >>> 
    >>> # 5. 벡터 유사도 검색
    >>> query_embedding = [0.15, 0.25, ...]  # 384차원
    >>> results = db_manager.search_similar_chunks(
    ...     query_embedding=query_embedding,
    ...     top_k=5
    ... )
    >>> 
    >>> for chunk, similarity in results:
    ...     print(f"Similarity: {similarity:.2f}")
    ...     print(f"Code: {chunk.content}")

데이터베이스 스키마:
    1. repositories
       - id, url, name, owner, description, language, stars
       - created_at, updated_at
    
    2. code_chunks
       - id, repository_id (FK), file_path, language
       - content, embedding (vector 384), chunk_index
       - start_line, end_line, created_at
    
    3. analysis_results
       - id, repository_id (FK), requirement
       - is_executable, reasoning, confidence_score
       - ai_model, code_context, metadata
       - created_at

pgvector 설정:
    - embedding 컬럼: vector(384) 타입
    - HNSW 인덱스: 빠른 유사도 검색
    - 코사인 유사도: vector_cosine_ops

트랜잭션 관리:
    DatabaseManager는 컨텍스트 매니저를 사용한 안전한 트랜잭션을 제공합니다.
    
    >>> with db_manager.get_session() as session:
    ...     # 자동 커밋/롤백
    ...     repo = session.query(Repository).first()
"""

from .models import Base, Repository, CodeChunk, AnalysisResult
from .db_manager import DatabaseManager

__all__ = [
    # Models
    "Base",
    "Repository",
    "CodeChunk",
    "AnalysisResult",
    
    # Manager
    "DatabaseManager",
]

# 버전 정보
__version__ = "1.0.0"

# 패키지 레벨 독스트링
__doc__ = """
데이터베이스 모듈 - PostgreSQL + pgvector

이 모듈은 SQLAlchemy ORM과 pgvector를 사용하여 코드 저장 및 벡터 검색을 제공합니다.

핵심 기능:
1. GitHub 저장소 정보 관리
2. 코드 청크 + 벡터 임베딩 저장
3. AI 분석 결과 저장
4. 벡터 유사도 검색 (HNSW 인덱스)
5. 트랜잭션 관리

사용 흐름:
DatabaseManager(url) → create_tables() → CRUD 작업

최소 사용 예시:
    from config import settings
    from src.database import DatabaseManager
    
    db = DatabaseManager(settings.database_url)
    db.create_tables()
    
    # 저장소 생성
    repo = db.create_repository(
        url="https://github.com/user/repo",
        name="repo"
    )
    
    # 코드 청크 생성
    chunk = db.create_code_chunk(
        repository_id=repo.id,
        file_path="main.py",
        content="def main(): pass",
        embedding=[0.1] * 384
    )
    
    # 유사도 검색
    results = db.search_similar_chunks(
        query_embedding=[0.15] * 384,
        top_k=5
    )

필수 환경:
- PostgreSQL 12+
- pgvector extension
- SQLAlchemy 2.0+

설치:
    # PostgreSQL (Docker)
    docker run -d \\
        -e POSTGRES_PASSWORD=password \\
        -e POSTGRES_DB=probe_db \\
        -p 5432:5432 \\
        pgvector/pgvector:pg16
    
    # Python 패키지
    pip install sqlalchemy psycopg2-binary pgvector
"""