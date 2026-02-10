"""
AWS Bedrock 임베딩 생성기

AWS Bedrock Titan을 사용하여 텍스트(코드)를 벡터로 변환합니다.

주요 기능:
- 코드를 1024차원 벡터로 변환 (Titan v2 기본)
- 배치 처리 지원
- LangChain 호환 인터페이스
"""

from typing import List, Union, Optional
import boto3
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BedrockEmbeddings:
    """
    AWS Bedrock Titan 임베딩 생성기
    
    Bedrock Titan 모델을 사용하여 코드를 벡터로 변환합니다.
    
    Attributes:
        client: Bedrock Runtime 클라이언트
        model_id: Titan 임베딩 모델 ID
        dimension: 벡터 차원 (512 또는 1024)
        region: AWS 리전
    
    Example:
        >>> embeddings = BedrockEmbeddings()
        >>> vector = embeddings.generate("def login(): pass")
        >>> print(vector.shape)  # (1024,)
        >>> 
        >>> # 배치 처리
        >>> vectors = embeddings.generate_batch([
        ...     "def login(): pass",
        ...     "def logout(): pass"
        ... ])
        >>> print(vectors.shape)  # (2, 1024)
    """
    
    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dimension: int = 1024,
        normalize_embeddings: bool = True
    ):
        """
        BedrockEmbeddings 초기화
        
        Args:
            model_id: Titan 임베딩 모델 ID
                - amazon.titan-embed-text-v2:0 (최신, 1024차원)
                - amazon.titan-embed-text-v1 (구버전, 1536차원)
            region: AWS 리전
            dimension: 벡터 차원 (512 또는 1024, v2만 지원)
            normalize_embeddings: 벡터 정규화 여부
        
        Note:
            AWS 인증은 IAM Role 또는 환경변수를 사용합니다.
        """
        self.model_id = model_id
        self.region = region
        self.dimension = dimension
        self.normalize_embeddings = normalize_embeddings
        
        try:
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region
            )
            logger.info(
                f"BedrockEmbeddings initialized: "
                f"model={model_id}, region={region}, dim={dimension}"
            )
            
            # 연결 테스트
            self._test_connection()
            
        except Exception as e:
            error_msg = f"Failed to initialize Bedrock embeddings: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        
        # 통계
        self._embedding_count = 0
    
    def _test_connection(self) -> None:
        """Bedrock 연결 테스트"""
        try:
            test_text = "test"
            self.generate(test_text)
            logger.info("Bedrock embeddings connection test successful")
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise
    
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
            >>> embeddings = BedrockEmbeddings()
            >>> code = "def login(username, password): return True"
            >>> vector = embeddings.generate(code)
            >>> print(vector.shape)  # (1024,)
        """
        if not text or not text.strip():
            logger.warning("Empty text provided, returning zero vector")
            return np.zeros(self.dimension, dtype=np.float32)
        
        try:
            # 정규화 설정
            norm = normalize if normalize is not None else self.normalize_embeddings
            
            # Titan 임베딩 요청
            body = {
                "inputText": text,
                "dimensions": self.dimension,
                "normalize": norm
            }
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            # 응답 파싱
            response_body = json.loads(response['body'].read())
            embedding = response_body['embedding']
            
            self._embedding_count += 1
            
            return np.array(embedding, dtype=np.float32)
            
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
        여러 텍스트의 임베딩을 배치로 생성
        
        Note:
            Bedrock은 현재 배치 API를 제공하지 않으므로,
            순차적으로 처리합니다. (향후 개선 가능)
        
        Args:
            texts: 텍스트 리스트
            batch_size: 배치 크기 (현재 미사용, 호환성 유지)
            show_progress: 진행 상황 표시 여부
            normalize: 정규화 여부
        
        Returns:
            np.ndarray: 임베딩 행렬 (shape: (len(texts), dimension))
        
        Example:
            >>> embeddings = BedrockEmbeddings()
            >>> codes = [
            ...     "def login(): pass",
            ...     "def logout(): pass"
            ... ]
            >>> vectors = embeddings.generate_batch(codes)
            >>> print(vectors.shape)  # (2, 1024)
        """
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])
        
        embeddings_list = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            try:
                embedding = self.generate(text, normalize=normalize)
                embeddings_list.append(embedding)
                
                if show_progress and (i + 1) % 10 == 0:
                    logger.info(f"Embedded {i + 1}/{total} texts")
                    
            except Exception as e:
                logger.warning(f"Failed to embed text {i}: {e}")
                # 실패한 경우 0 벡터 추가
                embeddings_list.append(np.zeros(self.dimension, dtype=np.float32))
        
        logger.info(f"Generated {len(embeddings_list)} embeddings")
        
        return np.array(embeddings_list, dtype=np.float32)
    
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
        """
        if isinstance(text, str):
            return self.generate(text, **kwargs)
        elif isinstance(text, list):
            return self.generate_batch(text, **kwargs)
        else:
            raise TypeError(f"Expected str or List[str], got {type(text)}")
    
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
        """
        return {
            "model_id": self.model_id,
            "dimension": self.dimension,
            "region": self.region,
            "normalize_embeddings": self.normalize_embeddings,
            "embeddings_generated": self._embedding_count,
            "provider": "AWS Bedrock"
        }
    
    def __repr__(self) -> str:
        return (
            f"BedrockEmbeddings("
            f"model='{self.model_id}', "
            f"dimension={self.dimension}, "
            f"region='{self.region}')"
        )
    
    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        """함수 호출 인터페이스"""
        return self.encode(text)


# 사용 예시
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        print("\n=== Test 1: Basic Embedding ===")
        embeddings = BedrockEmbeddings()
        
        code = "def login(username, password): return True"
        vector = embeddings.generate(code)
        
        print(f"Code: {code}")
        print(f"Vector shape: {vector.shape}")
        print(f"Vector (first 5): {vector[:5]}")
        
        print("\n=== Test 2: Batch Embedding ===")
        codes = [
            "def login(): pass",
            "def logout(): pass",
            "def register(): pass"
        ]
        
        vectors = embeddings.generate_batch(codes, show_progress=True)
        print(f"Generated {len(vectors)} vectors")
        print(f"Shape: {vectors.shape}")
        
        print("\n=== Test 3: Similarity ===")
        text1 = "def login(user, password): return True"
        text2 = "def authenticate(username, pwd): return True"
        
        sim = embeddings.compute_similarity(text1, text2)
        print(f"Similarity: {sim:.3f}")
        
        print("\n=== Test 4: Model Info ===")
        info = embeddings.get_model_info()
        for key, value in info.items():
            print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. AWS credentials are configured")
        print("2. Bedrock is available in your region")
        print("3. You have access to Titan models")