"""
LangChain 통합 모듈

이 모듈은 LangChain을 사용하여 GitHub 코드 수집, 텍스트 분할, RAG 파이프라인을 제공.

주요 컴포넌트:
- GitHubCodeLoader: GitHub 저장소에서 코드 파일 다운로드
- CodeChunker: 코드를 의미 있는 청크로 분할
- RAGPipeline: 벡터 검색 및 컨텍스트 검색

사용 예시:
    >>> from src.langchain_integration import GitHubCodeLoader, CodeChunker, RAGPipeline
    >>> 
    >>> # 1. GitHub에서 코드 로드
    >>> loader = GitHubCodeLoader()
    >>> docs = loader.load_repository("https://github.com/user/repo")
    >>> 
    >>> # 2. 코드 청킹
    >>> chunker = CodeChunker()
    >>> chunks = chunker.split_documents(docs, language="python")
    >>> 
    >>> # 3. RAG 파이프라인으로 검색
    >>> rag = RAGPipeline()
    >>> rag.add_documents(chunks)
    >>> results = rag.search_similar("로그인 기능", top_k=5)
"""

from .github_loader import GitHubCodeLoader
from .text_splitter import CodeChunker
from .rag_pipeline import RAGPipeline

__all__ = [
    "GitHubCodeLoader",
    "CodeChunker",
    "RAGPipeline",
]

# 버전 정보
__version__ = "1.0.0"