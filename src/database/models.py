"""
데이터베이스 모델 정의

SQLAlchemy ORM을 사용한 테이블 모델 정의
- Repository: GitHub 저장소 정보
- CodeChunk: 코드 조각 + 벡터 임베딩
- AnalysisResult: AI 분석 결과
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Float
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Repository(Base):
    """
    GitHub 저장소 정보
    
    Attributes:
        id: 기본 키
        url: GitHub 저장소 URL
        name: 저장소 이름
        owner: 저장소 소유자
        description: 저장소 설명
        language: 주 프로그래밍 언어
        stars: 스타 개수
        created_at: 레코드 생성 시각
        updated_at: 레코드 수정 시각
        code_chunks: 연관된 코드 청크들
        analysis_results: 연관된 분석 결과들
    """
    
    __tablename__ = "repositories"
    
    # 기본 키
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 저장소 정보
    url = Column(String(500), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    owner = Column(String(200))
    description = Column(Text)
    language = Column(String(50))
    stars = Column(Integer, default=0)
    
    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 관계 (1:N)
    code_chunks = relationship(
        "CodeChunk",
        back_populates="repository",
        cascade="all, delete-orphan"
    )
    test_scripts = relationship(
        "TestScript",
        back_populates="repository",
        cascade="all, delete-orphan"
    )
    
    def __repr__(self):
        return f"<Repository(id={self.id}, name='{self.name}', url='{self.url}')>"
    
    def to_dict(self):
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            "id": self.id,
            "url": self.url,
            "name": self.name,
            "owner": self.owner,
            "description": self.description,
            "language": self.language,
            "stars": self.stars,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class CodeChunk(Base):
    """
    코드 청크 + 벡터 임베딩
    
    Attributes:
        id: 기본 키
        repository_id: 연관된 저장소 ID (외래 키)
        file_path: 파일 경로
        language: 프로그래밍 언어
        content: 코드 내용
        embedding: 벡터 임베딩 (384차원)
        chunk_index: 청크 순서
        start_line: 시작 라인 번호
        end_line: 종료 라인 번호
        created_at: 레코드 생성 시각
        repository: 연관된 저장소 객체
    """
    
    __tablename__ = "code_chunks"
    
    # 기본 키
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 외래 키
    repository_id = Column(
        Integer,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # 파일 정보
    file_path = Column(String(500), nullable=False)
    language = Column(String(50))
    
    # 코드 내용
    content = Column(Text, nullable=False)
    
    # 벡터 임베딩 (384차원 - all-MiniLM-L6-v2)
    embedding = Column(Vector(384))
    
    # 청크 정보
    chunk_index = Column(Integer, default=0)
    start_line = Column(Integer)
    end_line = Column(Integer)
    
    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # 관계 (N:1)
    repository = relationship("Repository", back_populates="code_chunks")
    
    def __repr__(self):
        return (
            f"<CodeChunk(id={self.id}, "
            f"repo_id={self.repository_id}, "
            f"file='{self.file_path}', "
            f"chunk={self.chunk_index})>"
        )
    
    def to_dict(self):
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            "id": self.id,
            "repository_id": self.repository_id,
            "file_path": self.file_path,
            "language": self.language,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "has_embedding": self.embedding is not None
        }


class TestScript(Base):
    """
    생성된 Playwright 테스트 스크립트
    
    Attributes:
        id: 기본 키
        repository_id: 연관된 저장소 ID (외래 키)
        requirement: 테스트 요구사항
        test_type: 테스트 타입 (e2e, unit, integration)
        base_url: 테스트 대상 URL
        test_code: 생성된 Playwright 코드
        filename: 파일명 (예: login_20250109.spec.js)
        test_description: 테스트 설명
        test_cases: 테스트 케이스 목록 (JSON)
        lines_of_code: 코드 라인 수
        storage_url: S3/MinIO 저장 URL
        download_url: 다운로드 URL (Pre-signed)
        ai_model: 사용된 AI 모델 (gpt, claude)
        code_context: 생성에 사용된 코드 컨텍스트
        metadata: 추가 메타데이터 (JSON)
        created_at: 레코드 생성 시각
        repository: 연관된 저장소 객체
    """
    
    __tablename__ = "test_scripts"
    
    # 기본 키
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 외래 키
    repository_id = Column(
        Integer,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # 테스트 요구사항
    requirement = Column(Text, nullable=False)
    test_type = Column(String(50), default="e2e")  # e2e, unit, integration
    base_url = Column(String(500))  # 테스트 대상 URL
    
    # 생성된 코드
    test_code = Column(Text, nullable=False)
    filename = Column(String(200))  # login_20250109.spec.js
    
    # 테스트 메타데이터
    test_description = Column(Text)
    test_cases = Column(JSON, default=[])  # ["정상 로그인", "실패 케이스"]
    lines_of_code = Column(Integer)
    
    # 저장 위치
    storage_url = Column(String(500))  # s3://bucket/path
    download_url = Column(String(1000))  # Pre-signed URL
    
    # AI 모델 정보
    ai_model = Column(String(50))  # "gpt", "claude"
    
    # 생성 컨텍스트
    code_context = Column(Text)  # RAG에서 검색된 코드
    
    # 추가 정보 (JSON)
    metadata = Column(JSON, default={})
    
    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # 관계 (N:1)
    repository = relationship("Repository", back_populates="test_scripts")
    
    def __repr__(self):
        return (
            f"<TestScript(id={self.id}, "
            f"repo_id={self.repository_id}, "
            f"filename='{self.filename}', "
            f"type='{self.test_type}')>"
        )
    
    def to_dict(self):
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            "id": self.id,
            "repository_id": self.repository_id,
            "requirement": self.requirement,
            "test_type": self.test_type,
            "base_url": self.base_url,
            "filename": self.filename,
            "test_description": self.test_description,
            "test_cases": self.test_cases,
            "lines_of_code": self.lines_of_code,
            "storage_url": self.storage_url,
            "download_url": self.download_url,
            "ai_model": self.ai_model,
            "code_preview": self.test_code[:200] + "..." if self.test_code else None,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# 인덱스 생성 (성능 최적화)
# pgvector를 위한 HNSW 인덱스는 별도로 생성 필요
# CREATE INDEX ON code_chunks USING hnsw (embedding vector_cosine_ops);

if __name__ == "__main__":
    # 테이블 구조 출력
    print("\n=== Database Models ===\n")
    
    print("1. Repository Table:")
    print(f"   Columns: {[c.name for c in Repository.__table__.columns]}")
    
    print("\n2. CodeChunk Table:")
    print(f"   Columns: {[c.name for c in CodeChunk.__table__.columns]}")
    
    print("\n3. AnalysisResult Table:")
    print(f"   Columns: {[c.name for c in AnalysisResult.__table__.columns]}")
    
    print("\n=== Relationships ===")
    print("Repository → CodeChunk: 1:N")
    print("Repository → AnalysisResult: 1:N")