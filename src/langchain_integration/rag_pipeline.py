"""
RAG (Retrieval-Augmented Generation) Pipeline

AWS Bedrock + FAISS를 사용한 RAG 파이프라인
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import logging

from src.embeddings import BedrockEmbeddings

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG 파이프라인 (Bedrock + FAISS)
    
    코드 청크를 FAISS 벡터 DB에 저장하고,
    요구사항과 유사한 코드를 검색합니다.
    
    특징:
    - AWS Bedrock Titan 임베딩 (1024차원)
    - FAISS 메모리 기반 벡터 스토어 (빠름, 임시)
    - Tree-First 전략 지원
    
    Attributes:
        vector_store: FAISS 벡터 저장소
        embeddings: Bedrock 임베딩 생성기
        index_name: 인덱스 이름
    """
    
    def __init__(
        self,
        index_name: str = "code_embeddings",
        region: str = "us-east-1",
        dimension: int = 1024
    ):
        """
        RAG 파이프라인 초기화
        
        Args:
            index_name: FAISS 인덱스 이름
            region: AWS 리전
            dimension: 임베딩 차원 (512 또는 1024)
        """
        self.index_name = index_name
        self.region = region
        self.dimension = dimension
        
        # Bedrock 임베딩 생성기 초기화
        self.embeddings = BedrockEmbeddings(
            region=region,
            dimension=dimension
        )
        
        # FAISS 벡터 저장소 (초기에는 None, 문서 추가 시 생성)
        self.vector_store: Optional[FAISS] = None
        
        logger.info(
            f"RAGPipeline initialized: "
            f"index={index_name}, region={region}, dim={dimension}"
        )
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 32
    ) -> List[str]:
        """
        문서(코드 청크)를 벡터 DB에 추가
        
        Args:
            documents: LangChain Document 객체 리스트
            batch_size: 배치 크기 (임베딩 생성 시)
        
        Returns:
            List[str]: 추가된 문서의 ID 리스트
        """
        try:
            if not documents:
                logger.warning("No documents to add")
                return []
            
            # FAISS 벡터 저장소 생성 또는 업데이트
            if self.vector_store is None:
                # 처음 생성
                logger.info(f"Creating new FAISS index with {len(documents)} documents")
                self.vector_store = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            else:
                # 기존에 추가
                logger.info(f"Adding {len(documents)} documents to existing index")
                self.vector_store.add_documents(documents)
            
            logger.info(f"Successfully added {len(documents)} documents to FAISS")
            
            # Document ID 생성 (FAISS는 자동 ID 생성하지 않음)
            ids = [f"doc_{i}" for i in range(len(documents))]
            return ids
        
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        유사도 검색 (Similarity Search)
        
        Args:
            query: 검색 쿼리 (요구사항)
            top_k: 반환할 문서 개수
            score_threshold: 최소 유사도 점수 (낮을수록 유사)
        
        Returns:
            List[Document]: 유사한 문서 리스트
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty, no documents to search")
            return []
        
        try:
            logger.debug(f"Searching for: '{query}' (top_k={top_k})")
            
            if score_threshold is not None:
                # 점수 기반 검색 (FAISS는 거리 반환, 낮을수록 유사)
                results_with_scores = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k
                )
                # 임계값 필터링 (거리가 작을수록 유사하므로 <= 사용)
                filtered_results = [
                    doc for doc, score in results_with_scores 
                    if score <= score_threshold
                ]
                logger.info(
                    f"Found {len(filtered_results)} documents "
                    f"(threshold: {score_threshold})"
                )
                return filtered_results
            else:
                # 일반 유사도 검색
                results = self.vector_store.similarity_search(
                    query=query,
                    k=top_k
                )
                logger.info(f"Found {len(results)} similar documents")
                return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def search_with_scores(
        self,
        query: str,
        top_k: int = 5
    ) -> List[tuple[Document, float]]:
        """
        유사도 점수와 함께 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 문서 개수
        
        Returns:
            List[tuple[Document, float]]: (문서, 거리 점수) 튜플 리스트
                                          (낮을수록 유사함)
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty")
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            logger.info(
                f"Found {len(results)} documents with scores for query: '{query}'"
            )
            return results
        
        except Exception as e:
            logger.error(f"Search with scores failed: {e}")
            raise
    
    def retrieve_context(
        self,
        requirement: str,
        top_k: int = 5,
        max_chars: int = 4000,
        file_tree: Optional[str] = None
    ) -> str:
        """
        RAG용 컨텍스트 검색 (AI 입력용)
        
        요구사항과 관련된 코드를 검색하여 문자열로 반환합니다.
        Tree-First 전략 지원.
        
        Args:
            requirement: 요구사항 설명
            top_k: 검색할 청크 개수
            max_chars: 최대 문자 수 (토큰 제한)
            file_tree: 전체 파일 구조 (Tree-First 전략용)
        
        Returns:
            str: 검색된 코드 컨텍스트 (AI 입력용)
        """
        try:
            context_parts = []
            
            # 1. Tree-First 전략: 파일 구조 먼저 제공
            if file_tree:
                tree_section = f"""
=== Project File Structure ===
{file_tree}

=== Relevant Code Chunks ===
"""
                context_parts.append(tree_section)
            
            # 2. 유사 코드 검색
            docs = self.search_similar(query=requirement, top_k=top_k)
            
            # 3. 컨텍스트 문자열 생성
            total_chars = len(context_parts[0]) if context_parts else 0
            
            for i, doc in enumerate(docs):
                # 메타데이터 추출
                file_path = doc.metadata.get("source", "unknown")
                language = doc.metadata.get("language", "unknown")
                
                # 코드 청크 포맷팅
                chunk_text = f"""
--- Code Chunk {i+1} ---
File: {file_path}
Language: {language}

{doc.page_content}

"""
                # 최대 문자 수 제한
                if total_chars + len(chunk_text) > max_chars:
                    logger.warning(
                        f"Context truncated at {total_chars} chars "
                        f"(limit: {max_chars})"
                    )
                    break
                
                context_parts.append(chunk_text)
                total_chars += len(chunk_text)
            
            context = "\n".join(context_parts)
            
            logger.info(
                f"Retrieved context: {len(docs)} chunks, "
                f"{total_chars} chars for '{requirement}'"
            )
            
            return context
        
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            raise
    
    def save_index(self, filepath: str) -> None:
        """FAISS 인덱스를 디스크에 저장"""
        if self.vector_store is None:
            logger.warning("No vector store to save")
            return
        
        try:
            self.vector_store.save_local(filepath)
            logger.info(f"FAISS index saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self, filepath: str) -> None:
        """FAISS 인덱스를 디스크에서 로드"""
        try:
            self.vector_store = FAISS.load_local(
                filepath,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS index loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def clear(self) -> None:
        """벡터 저장소 초기화 (메모리 해제)"""
        self.vector_store = None
        logger.info("Vector store cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """벡터 저장소 통계 정보"""
        try:
            if self.vector_store is None:
                return {
                    "index_name": self.index_name,
                    "region": self.region,
                    "dimension": self.dimension,
                    "is_initialized": False,
                    "document_count": 0
                }
            
            # FAISS 인덱스 크기
            index = self.vector_store.index
            doc_count = index.ntotal if hasattr(index, 'ntotal') else 0
            
            return {
                "index_name": self.index_name,
                "region": self.region,
                "dimension": self.dimension,
                "is_initialized": True,
                "document_count": doc_count
            }
        
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "index_name": self.index_name,
                "error": str(e)
            }
