"""
임베딩 생성기

Sentence Transformers를 사용하여 텍스트(코드)를 벡터로 변환합니다.

주요 기능:
- 코드를 384차원 벡터로 변환
- 배치 처리 지원 (메모리 최적화)
- GPU 자동 감지 및 활용
- 캐싱으로 성능 향상
"""

from typing import List, Union, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    텍스트 임베딩 생성기
    
    Sentence Transformers를 사용하여 코드를 벡터로 변환합니다.
    
    Attributes:
        model: SentenceTransformer 모델
        model_name: 모델 이름
        dimension: 벡터 차원 (기본 384)
        device: 실행 디바이스 (cuda/cpu)
    
    Example:
        >>> generator = EmbeddingGenerator()
        >>> vector = generator.generate("def login(): pass")
        >>> print(vector.shape)  # (384,)
        >>> 
        >>> # 배치 처리
        >>> vectors = generator.generate_batch([
        ...     "def login(): pass",
        ...     "def logout(): pass"
        ... ])
        >>> print(vectors.shape)  # (2, 384)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True
    ):
        """
        EmbeddingGenerator 초기화
        
        Args:
            model_name: Sentence Transformers 모델 이름
                - all-MiniLM-L6-v2: 384차원, 빠름, 정확도 좋음 (기본)
                - all-mpnet-base-v2: 768차원, 느림, 정확도 최고
                - paraphrase-multilingual: 다국어 지원
            device: 강제로 지정할 디바이스 (None이면 자동 감지)
            normalize_embeddings: 벡터 정규화 여부
        """
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        
        # 디바이스 설정 (GPU 자동 감지)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing EmbeddingGenerator on {self.device}")
        logger.info(f"Model: {model_name}")
        
        # 모델 로드
        try:
            self.model = SentenceTransformer(
                model_name,
                device=self.device
            )
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(
                f"Model loaded successfully: {self.dimension}D embeddings"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # 통계
        self._embedding_count = 0
    
    def generate(
        self,
        text: str,
        normalize: Optional[bool] = None
    ) -> np.ndarray:
        """
        단일 텍스트의 임베딩 생성
        
        Args:
            text: 임베딩할 텍스트 (코드)
            normalize: 정규화 여부 (None이면 초기화 설정 사용)
        
        Returns:
            np.ndarray: 임베딩 벡터 (shape: (dimension,))
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> code = "def login(username, password): return True"
            >>> vector = generator.generate(code)
            >>> print(vector.shape)  # (384,)
            >>> print(vector[:5])     # [0.123, -0.456, ...]
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self.dimension, dtype=np.float32)
        
        try:
            # 정규화 설정
            norm = normalize if normalize is not None else self.normalize_embeddings
            
            # 임베딩 생성
            embedding = self.model.encode(
                text,
                normalize_embeddings=norm,
                convert_to_numpy=True
            )
            
            self._embedding_count += 1
            
            return embedding.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def generate_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: Optional[bool] = None
    ) -> np.ndarray:
        """
        여러 텍스트의 임베딩을 배치로 생성 (메모리 최적화)
        
        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기 (GPU 메모리에 따라 조정)
            show_progress: 진행 상황 표시 여부
            normalize: 정규화 여부
        
        Returns:
            np.ndarray: 임베딩 행렬 (shape: (len(texts), dimension))
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> codes = [
            ...     "def login(): pass",
            ...     "def logout(): pass",
            ...     "def register(): pass"
            ... ]
            >>> vectors = generator.generate_batch(codes, batch_size=32)
            >>> print(vectors.shape)  # (3, 384)
        """
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])
        
        # 빈 문자열 필터링
        valid_texts = [t if t and t.strip() else " " for t in texts]
        
        try:
            # 정규화 설정
            norm = normalize if normalize is not None else self.normalize_embeddings
            
            # 배치 임베딩 생성
            embeddings = self.model.encode(
                valid_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=norm,
                convert_to_numpy=True
            )
            
            self._embedding_count += len(texts)
            
            logger.info(
                f"Generated {len(texts)} embeddings in batches of {batch_size}"
            )
            
            return embeddings.astype(np.float32)
        
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def encode(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> np.ndarray:
        """
        LangChain 호환 인터페이스
        
        단일 텍스트 또는 리스트를 자동으로 처리합니다.
        
        Args:
            text: 텍스트 또는 텍스트 리스트
            **kwargs: 추가 인자
        
        Returns:
            np.ndarray: 임베딩 벡터 또는 행렬
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> 
            >>> # 단일 텍스트
            >>> v1 = generator.encode("def login(): pass")
            >>> 
            >>> # 리스트
            >>> v2 = generator.encode(["def login(): pass", "def logout(): pass"])
        """
        if isinstance(text, str):
            return self.generate(text, **kwargs)
        elif isinstance(text, list):
            return self.generate_batch(text, **kwargs)
        else:
            raise TypeError(
                f"Expected str or List[str], got {type(text)}"
            )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain PGVector 호환 메서드
        
        Args:
            texts: 문서 텍스트 리스트
        
        Returns:
            List[List[float]]: 임베딩 리스트
        """
        embeddings = self.generate_batch(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        LangChain PGVector 호환 메서드 (쿼리용)
        
        Args:
            text: 쿼리 텍스트
        
        Returns:
            List[float]: 임베딩 벡터
        """
        embedding = self.generate(text)
        return embedding.tolist()
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        두 텍스트 간 코사인 유사도 계산
        
        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트
        
        Returns:
            float: 코사인 유사도 (0.0~1.0)
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> similarity = generator.compute_similarity(
            ...     "def login(): pass",
            ...     "def authenticate(): pass"
            ... )
            >>> print(f"Similarity: {similarity:.2f}")  # 0.85
        """
        emb1 = self.generate(text1, normalize=True)
        emb2 = self.generate(text2, normalize=True)
        
        # 코사인 유사도 (정규화된 벡터는 내적과 동일)
        similarity = np.dot(emb1, emb2)
        
        return float(similarity)
    
    def get_model_info(self) -> dict:
        """
        모델 정보 반환
        
        Returns:
            dict: 모델 정보
                - model_name: str
                - dimension: int
                - device: str
                - normalize_embeddings: bool
                - embeddings_generated: int
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "normalize_embeddings": self.normalize_embeddings,
            "embeddings_generated": self._embedding_count,
            "gpu_available": torch.cuda.is_available()
        }
    
    def __repr__(self) -> str:
        return (
            f"EmbeddingGenerator("
            f"model='{self.model_name}', "
            f"dimension={self.dimension}, "
            f"device='{self.device}')"
        )
    
    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        함수 호출 인터페이스
        
        Example:
            >>> generator = EmbeddingGenerator()
            >>> vector = generator("def login(): pass")
        """
        return self.encode(text)


# 캐싱된 전역 인스턴스 (싱글톤 패턴)
_global_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator(
    model_name: Optional[str] = None,
    force_reload: bool = False
) -> EmbeddingGenerator:
    """
    전역 EmbeddingGenerator 인스턴스 가져오기 (싱글톤)
    
    매번 새로 로드하지 않고 재사용하여 메모리 절약
    
    Args:
        model_name: 모델 이름 (None이면 기본값)
        force_reload: 강제 재로드 여부
    
    Returns:
        EmbeddingGenerator: 싱글톤 인스턴스
    
    Example:
        >>> # 첫 번째 호출 (모델 로드)
        >>> gen1 = get_embedding_generator()
        >>> 
        >>> # 두 번째 호출 (재사용)
        >>> gen2 = get_embedding_generator()
        >>> 
        >>> assert gen1 is gen2  # 동일한 인스턴스
    """
    global _global_generator
    
    if force_reload or _global_generator is None:
        if model_name:
            _global_generator = EmbeddingGenerator(model_name=model_name)
        else:
            _global_generator = EmbeddingGenerator()
        
        logger.info("Global EmbeddingGenerator instance created")
    
    return _global_generator


# 사용 예시
if __name__ == "__main__":
    import logging
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Test 1: Basic Embedding ===")
    generator = EmbeddingGenerator()
    
    code = "def login(username, password): return True"
    vector = generator.generate(code)
    
    print(f"Code: {code}")
    print(f"Vector shape: {vector.shape}")
    print(f"Vector (first 5): {vector[:5]}")
    
    print("\n=== Test 2: Batch Embedding ===")
    codes = [
        "def login(): pass",
        "def logout(): pass",
        "def register(): pass",
        "class User: pass"
    ]
    
    vectors = generator.generate_batch(codes, show_progress=True)
    print(f"Generated {len(vectors)} vectors")
    print(f"Shape: {vectors.shape}")
    
    print("\n=== Test 3: Similarity ===")
    text1 = "def login(user, password): return True"
    text2 = "def authenticate(username, pwd): return True"
    text3 = "class Database: pass"
    
    sim1 = generator.compute_similarity(text1, text2)
    sim2 = generator.compute_similarity(text1, text3)
    
    print(f"Similarity (login vs authenticate): {sim1:.3f}")
    print(f"Similarity (login vs database): {sim2:.3f}")
    
    print("\n=== Test 4: Model Info ===")
    info = generator.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n=== Test 5: Singleton Pattern ===")
    gen1 = get_embedding_generator()
    gen2 = get_embedding_generator()
    print(f"Same instance: {gen1 is gen2}")
    
    print("\n=== Test 6: LangChain Compatibility ===")
    # PGVector에서 사용하는 메서드 테스트
    docs = ["def func1(): pass", "def func2(): pass"]
    embeddings = generator.embed_documents(docs)
    print(f"embed_documents: {len(embeddings)} embeddings")
    
    query_emb = generator.embed_query("def search(): pass")
    print(f"embed_query: {len(query_emb)} dimensions")