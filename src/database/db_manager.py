"""
데이터베이스 관리자

데이터베이스 연결, CRUD 작업, 트랜잭션 관리를 담당합니다.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging

from .models import Base, Repository, CodeChunk, AnalysisResult

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    데이터베이스 관리자
    
    SQLAlchemy를 사용한 데이터베이스 연결 및 CRUD 작업을 관리합니다.
    
    Attributes:
        engine: SQLAlchemy 엔진
        SessionLocal: 세션 팩토리
        database_url: 데이터베이스 연결 URL
    """
    
    def __init__(
        self,
        database_url: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        echo: bool = False
    ):
        """
        DatabaseManager 초기화
        
        Args:
            database_url: PostgreSQL 연결 URL
            pool_size: 커넥션 풀 크기
            max_overflow: 최대 오버플로우
            echo: SQL 로그 출력 여부
        """
        self.database_url = database_url
        
        # SQLAlchemy 엔진 생성
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            echo=echo,
            pool_pre_ping=True  # 연결 유효성 검사
        )
        
        # 세션 팩토리
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"DatabaseManager initialized: {database_url}")
    
    def create_tables(self) -> None:
        """
        모든 테이블 생성
        
        주의: 이미 테이블이 존재하면 스킵됩니다.
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
            
            # pgvector extension 확인
            self._ensure_pgvector_extension()
            
            # HNSW 인덱스 생성 (벡터 검색 최적화)
            self._create_vector_index()
        
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_tables(self) -> None:
        """
        모든 테이블 삭제 (주의: 데이터 손실!)
        """
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    def _ensure_pgvector_extension(self) -> None:
        """pgvector extension 활성화"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                conn.commit()
            logger.info("pgvector extension ensured")
        except Exception as e:
            logger.warning(f"Failed to create pgvector extension: {e}")
    
    def _create_vector_index(self) -> None:
        """
        벡터 검색을 위한 HNSW 인덱스 생성
        
        HNSW (Hierarchical Navigable Small World)는 빠른 유사도 검색을 위한 인덱스입니다.
        """
        try:
            with self.engine.connect() as conn:
                # 인덱스가 이미 있는지 확인
                result = conn.execute(text(
                    "SELECT 1 FROM pg_indexes "
                    "WHERE indexname = 'code_chunks_embedding_idx'"
                ))
                
                if not result.fetchone():
                    # HNSW 인덱스 생성
                    conn.execute(text(
                        "CREATE INDEX code_chunks_embedding_idx "
                        "ON code_chunks "
                        "USING hnsw (embedding vector_cosine_ops)"
                    ))
                    conn.commit()
                    logger.info("HNSW vector index created")
                else:
                    logger.info("HNSW vector index already exists")
        
        except Exception as e:
            logger.warning(f"Failed to create vector index: {e}")
    
    @contextmanager
    def get_session(self) -> Session:
        """
        데이터베이스 세션 컨텍스트 매니저
        
        Yields:
            Session: SQLAlchemy 세션
        
        Example:
            >>> with db_manager.get_session() as session:
            ...     repo = session.query(Repository).first()
            ...     print(repo.name)
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    # ===== Repository CRUD =====
    
    def create_repository(
        self,
        url: str,
        name: str,
        owner: Optional[str] = None,
        description: Optional[str] = None,
        language: Optional[str] = None,
        stars: int = 0
    ) -> Repository:
        """
        저장소 생성
        
        Args:
            url: GitHub 저장소 URL
            name: 저장소 이름
            owner: 소유자
            description: 설명
            language: 주 언어
            stars: 스타 개수
        
        Returns:
            Repository: 생성된 저장소 객체
        """
        with self.get_session() as session:
            repo = Repository(
                url=url,
                name=name,
                owner=owner,
                description=description,
                language=language,
                stars=stars
            )
            session.add(repo)
            session.flush()
            session.refresh(repo)
            
            logger.info(f"Repository created: {repo.name} (id={repo.id})")
            return repo
    
    def get_repository_by_url(self, url: str) -> Optional[Repository]:
        """URL로 저장소 조회"""
        with self.get_session() as session:
            repo = session.query(Repository).filter_by(url=url).first()
            if repo:
                session.expunge(repo)  # 세션에서 분리
            return repo
    
    def get_repository_by_id(self, repo_id: int) -> Optional[Repository]:
        """ID로 저장소 조회"""
        with self.get_session() as session:
            repo = session.query(Repository).filter_by(id=repo_id).first()
            if repo:
                session.expunge(repo)
            return repo
    
    def list_repositories(self, limit: int = 100) -> List[Repository]:
        """모든 저장소 조회"""
        with self.get_session() as session:
            repos = session.query(Repository).limit(limit).all()
            for repo in repos:
                session.expunge(repo)
            return repos
    
    # ===== CodeChunk CRUD =====
    
    def create_code_chunk(
        self,
        repository_id: int,
        file_path: str,
        content: str,
        embedding: List[float],
        language: Optional[str] = None,
        chunk_index: int = 0,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> CodeChunk:
        """
        코드 청크 생성
        
        Args:
            repository_id: 저장소 ID
            file_path: 파일 경로
            content: 코드 내용
            embedding: 벡터 임베딩 (384차원)
            language: 언어
            chunk_index: 청크 인덱스
            start_line: 시작 라인
            end_line: 종료 라인
        
        Returns:
            CodeChunk: 생성된 코드 청크
        """
        with self.get_session() as session:
            chunk = CodeChunk(
                repository_id=repository_id,
                file_path=file_path,
                content=content,
                embedding=embedding,
                language=language,
                chunk_index=chunk_index,
                start_line=start_line,
                end_line=end_line
            )
            session.add(chunk)
            session.flush()
            session.refresh(chunk)
            
            logger.debug(f"CodeChunk created: {file_path} chunk {chunk_index}")
            return chunk
    
    def get_chunks_by_repository(
        self,
        repository_id: int,
        limit: int = 1000
    ) -> List[CodeChunk]:
        """저장소의 모든 코드 청크 조회"""
        with self.get_session() as session:
            chunks = (
                session.query(CodeChunk)
                .filter_by(repository_id=repository_id)
                .limit(limit)
                .all()
            )
            for chunk in chunks:
                session.expunge(chunk)
            return chunks
    
    def search_similar_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        repository_id: Optional[int] = None
    ) -> List[tuple[CodeChunk, float]]:
        """
        벡터 유사도 검색
        
        Args:
            query_embedding: 쿼리 벡터
            top_k: 상위 K개 결과
            repository_id: 특정 저장소로 제한 (선택)
        
        Returns:
            List[tuple[CodeChunk, float]]: (청크, 유사도) 튜플 리스트
        """
        with self.get_session() as session:
            # 기본 쿼리
            query = session.query(
                CodeChunk,
                CodeChunk.embedding.cosine_distance(query_embedding).label("distance")
            )
            
            # 저장소 필터
            if repository_id:
                query = query.filter_by(repository_id=repository_id)
            
            # 유사도 순 정렬 및 제한
            results = query.order_by("distance").limit(top_k).all()
            
            # 세션에서 분리
            output = []
            for chunk, distance in results:
                session.expunge(chunk)
                similarity = 1 - distance  # 거리 → 유사도 변환
                output.append((chunk, similarity))
            
            return output
    
    # ===== AnalysisResult CRUD =====
    
    def create_analysis_result(
        self,
        repository_id: int,
        requirement: str,
        is_executable: bool,
        reasoning: str,
        confidence_score: int,
        ai_model: str = "gpt",
        code_context: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """
        분석 결과 생성
        
        Args:
            repository_id: 저장소 ID
            requirement: 요구사항
            is_executable: 실행 가능 여부
            reasoning: 판단 근거
            confidence_score: 신뢰도 (0-100)
            ai_model: AI 모델 이름
            code_context: 코드 컨텍스트
            metadata: 추가 메타데이터
        
        Returns:
            AnalysisResult: 생성된 분석 결과
        """
        with self.get_session() as session:
            result = AnalysisResult(
                repository_id=repository_id,
                requirement=requirement,
                is_executable=is_executable,
                reasoning=reasoning,
                confidence_score=confidence_score,
                ai_model=ai_model,
                code_context=code_context,
                metadata=metadata or {}
            )
            session.add(result)
            session.flush()
            session.refresh(result)
            
            logger.info(
                f"AnalysisResult created: repo_id={repository_id}, "
                f"executable={is_executable}, confidence={confidence_score}"
            )
            return result
    
    def get_analysis_results_by_repository(
        self,
        repository_id: int,
        limit: int = 100
    ) -> List[AnalysisResult]:
        """저장소의 모든 분석 결과 조회"""
        with self.get_session() as session:
            results = (
                session.query(AnalysisResult)
                .filter_by(repository_id=repository_id)
                .order_by(AnalysisResult.created_at.desc())
                .limit(limit)
                .all()
            )
            for result in results:
                session.expunge(result)
            return results
    
    def get_latest_analysis(
        self,
        repository_id: int,
        requirement: str
    ) -> Optional[AnalysisResult]:
        """특정 요구사항에 대한 최신 분석 결과 조회"""
        with self.get_session() as session:
            result = (
                session.query(AnalysisResult)
                .filter_by(repository_id=repository_id, requirement=requirement)
                .order_by(AnalysisResult.created_at.desc())
                .first()
            )
            if result:
                session.expunge(result)
            return result
    
    # ===== 유틸리티 =====
    
    def get_stats(self) -> Dict[str, int]:
        """데이터베이스 통계"""
        with self.get_session() as session:
            return {
                "repositories": session.query(Repository).count(),
                "code_chunks": session.query(CodeChunk).count(),
                "analysis_results": session.query(AnalysisResult).count()
            }
    
    def close(self) -> None:
        """데이터베이스 연결 종료"""
        self.engine.dispose()
        logger.info("Database connections closed")


# 사용 예시
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 테스트용 연결 (실제 DB 필요)
    db_url = "postgresql://probe_user:password@localhost:5432/probe_db"
    
    try:
        db_manager = DatabaseManager(db_url)
        
        print("\n=== Test 1: Create Tables ===")
        db_manager.create_tables()
        
        print("\n=== Test 2: Get Stats ===")
        stats = db_manager.get_stats()
        print(f"Stats: {stats}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("Note: Ensure PostgreSQL is running with pgvector extension")