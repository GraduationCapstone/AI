"""
임베딩 모듈

이 모듈은 텍스트(코드)를 벡터로 변환하는 임베딩 생성 기능을 제공합니다.

주요 컴포넌트:
- EmbeddingGenerator: Sentence Transformers 기반 임베딩 생성기
- get_embedding_generator: 싱글톤 패턴으로 전역 인스턴스 제공

사용 예시:
    >>> from src.embeddings import EmbeddingGenerator
    >>> 
    >>> # 1. Generator 생성
    >>> generator = EmbeddingGenerator()
    >>> 
    >>> # 2. 단일 텍스트 임베딩
    >>> code = "def login(username, password): return True"
    >>> vector = generator.generate(code)
    >>> print(vector.shape)  # (384,)
    >>> 
    >>> # 3. 배치 임베딩
    >>> codes = ["def login(): pass", "def logout(): pass"]
    >>> vectors = generator.generate_batch(codes)
    >>> print(vectors.shape)  # (2, 384)
    >>> 
    >>> # 4. 유사도 계산
    >>> similarity = generator.compute_similarity(
    ...     "def login(): pass",
    ...     "def authenticate(): pass"
    ... )
    >>> print(f"Similarity: {similarity:.2f}")

싱글톤 사용:
    >>> from src.embeddings import get_embedding_generator
    >>> 
    >>> # 매번 같은 인스턴스 반환 (메모리 절약)
    >>> gen1 = get_embedding_generator()
    >>> gen2 = get_embedding_generator()
    >>> assert gen1 is gen2

LangChain 호환:
    EmbeddingGenerator는 LangChain의 PGVector와 호환됩니다.
    
    >>> from langchain_community.vectorstores import PGVector
    >>> from src.embeddings import get_embedding_generator
    >>> 
    >>> embedding_function = get_embedding_generator()
    >>> vector_store = PGVector(
    ...     collection_name="code_embeddings",
    ...     connection_string="postgresql://...",
    ...     embedding_function=embedding_function
    ... )

모델 선택:
    기본 모델: all-MiniLM-L6-v2 (384차원, 빠름, 정확도 좋음)
    
    다른 모델 사용:
    >>> generator = EmbeddingGenerator(
    ...     model_name="sentence-transformers/all-mpnet-base-v2"  # 768차원
    ... )
    
    추천 모델:
    - all-MiniLM-L6-v2: 384차원, 빠름, 균형잡힘 (기본) ⭐
    - all-mpnet-base-v2: 768차원, 느림, 최고 정확도
    - paraphrase-multilingual-MiniLM-L12-v2: 다국어 지원

성능 최적화:
    - GPU 자동 감지 및 활용
    - 배치 처리로 대량 데이터 효율적 처리
    - 싱글톤 패턴으로 메모리 절약
    - 벡터 정규화로 유사도 계산 최적화
"""

from .embedding_generator import (
    EmbeddingGenerator,
    get_embedding_generator,
)

__all__ = [
    "EmbeddingGenerator",
    "get_embedding_generator",
]

# 버전 정보
__version__ = "1.0.0"

# 기본 설정
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DIMENSION = 384

# 패키지 레벨 독스트링
__doc__ = """
임베딩 모듈 - 텍스트를 벡터로 변환

이 모듈은 Sentence Transformers를 사용하여 코드를 벡터로 변환합니다.

핵심 기능:
1. 코드 → 벡터 변환 (384차원)
2. GPU 자동 활용
3. 배치 처리 (메모리 최적화)
4. LangChain PGVector 호환
5. 싱글톤 패턴 (메모리 절약)

사용 흐름:
EmbeddingGenerator() → generate() or generate_batch()

최소 사용 예시:
    from src.embeddings import EmbeddingGenerator
    
    generator = EmbeddingGenerator()
    vector = generator.generate("def login(): pass")
    
    print(vector.shape)  # (384,)

배치 처리 예시:
    codes = ["def func1(): pass", "def func2(): pass"]
    vectors = generator.generate_batch(codes, batch_size=32)
    
    print(vectors.shape)  # (2, 384)

LangChain 통합 예시:
    from langchain_community.vectorstores import PGVector
    from src.embeddings import get_embedding_generator
    
    embedding_function = get_embedding_generator()
    
    vector_store = PGVector(
        collection_name="code_embeddings",
        connection_string="postgresql://...",
        embedding_function=embedding_function
    )
    
    # 문서 추가
    vector_store.add_documents(documents)
    
    # 검색
    results = vector_store.similarity_search("로그인 기능", k=5)
"""