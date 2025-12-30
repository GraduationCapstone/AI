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
    analysis_results = relationship(
        "AnalysisResult",
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


class AnalysisResult(Base):
    """
    AI 분석 결과
    
    Attributes:
        id: 기본 키
        repository_id: 연관된 저장소 ID (외래 키)
        requirement: 사용자 요구사항
        is_executable: 실행 가능 여부
        reasoning: 판단 근거
        confidence_score: 신뢰도 점수 (0-100)
        ai_model: 사용된 AI 모델 (gpt, claude)
        code_context: 분석에 사용된 코드 컨텍스트
        metadata: 추가 메타데이터 (JSON)
        created_at: 레코드 생성 시각
        repository: 연관된 저장소 객체
    """
    
    __tablename__ = "analysis_results"
    
    # 기본 키
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # 외래 키
    repository_id = Column(
        Integer,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    
    # 요구사항
    requirement = Column(Text, nullable=False)
    
    # 분석 결과
    is_executable = Column(Boolean, nullable=False)
    reasoning = Column(Text, nullable=False)
    confidence_score = Column(Integer, nullable=False)  # 0-100
    
    # AI 모델 정보
    ai_model = Column(String(50))  # "gpt", "claude"
    
    # 분석 컨텍스트
    code_context = Column(Text)  # RAG에서 검색된 코드
    
    # 추가 정보 (JSON)
    metadata = Column(JSON, default={})
    
    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # 관계 (N:1)
    repository = relationship("Repository", back_populates="analysis_results")
    
    def __repr__(self):
        return (
            f"<AnalysisResult(id={self.id}, "
            f"repo_id={self.repository_id}, "
            f"executable={self.is_executable}, "
            f"confidence={self.confidence_score})>"
        )
    
    def to_dict(self):
        """딕셔너리로 변환 (JSON 직렬화용)"""
        return {
            "id": self.id,
            "repository_id": self.repository_id,
            "requirement": self.requirement,
            "is_executable": self.is_executable,
            "reasoning": self.reasoning,
            "confidence_score": self.confidence_score,
            "ai_model": self.ai_model,
            "code_context": self.code_context[:200] + "..." if self.code_context else None,
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