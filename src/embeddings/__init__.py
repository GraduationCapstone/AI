"""
임베딩 모듈

이 모듈은 AWS Bedrock Titan을 사용하여 텍스트(코드)를 벡터로 변환합니다.

주요 컴포넌트:
- BedrockEmbeddings: AWS Bedrock Titan 기반 (1024차원)

사용 예시:
    >>> from src.embeddings import BedrockEmbeddings
    >>> 
    >>> embeddings = BedrockEmbeddings(region="us-east-1")
    >>> vector = embeddings.generate("def login(): pass")
    >>> print(vector.shape)  # (1024,)

LangChain 호환:
    BedrockEmbeddings는 LangChain의 벡터 스토어와 호환됩니다.
    
    >>> from langchain_community.vectorstores import FAISS
    >>> from src.embeddings import BedrockEmbeddings
    >>> 
    >>> embedding_function = BedrockEmbeddings()
    >>> vector_store = FAISS.from_documents(
    ...     documents=documents,
    ...     embedding=embedding_function
    ... )
"""

from .bedrock_embeddings import BedrockEmbeddings

__all__ = ["BedrockEmbeddings"]

# 버전 정보
__version__ = "2.0.0"

# 기본 설정
DEFAULT_BEDROCK_MODEL = "amazon.titan-embed-text-v2:0"
DEFAULT_BEDROCK_DIMENSION = 1024