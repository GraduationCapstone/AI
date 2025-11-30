"""
RAG (Retrieval-Augmented Generation) Pipeline
"""

from typing import List, Dict, Any, Optional
from langchain.schema import Document
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import logging

from config import settings
from src.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG 파이프라인
    
    코드 청크를 벡터 DB에 저장하고, 
    요구사항과 유사한 코드를 검색합니다.
    
    Attributes:
        vector_store: PGVector 벡터 저장소
        embedding_generator: 임베딩 생성기
        collection_name: 벡터 컬렉션 이름
    """
    
    def __init__(
        self,
        collection_name: str = "code_embeddings",
        embedding_model: Optional[str] = None
    ):
        """
        RAG 파이프라인 초기화
        
        Args:
            collection_name: 벡터 컬렉션 이름 (기본값: code_embeddings)
            embedding_model: 임베딩 모델 이름 (기본값: settings.embedding_model)
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model or settings.embedding_model
        
        # 임베딩 생성기 초기화
        self.embedding_generator = EmbeddingGenerator(model_name=self.embedding_model)
        
        # PGVector 벡터 저장소 초기화
        self.vector_store = self._initialize_vector_store()
        
        logger.info(
            f"RAGPipeline initialized with collection '{collection_name}' "
            f"and model '{self.embedding_model}'"
        )
    
    def _initialize_vector_store(self) -> PGVector:
        """
        PGVector 벡터 저장소 초기화
        
        Returns:
            PGVector: 초기화된 벡터 저장소
        """
        try:
            # LangChain의 OpenAIEmbeddings를 커스텀 임베딩으로 래핑
            # (Sentence Transformers 사용)
            embedding_function = self.embedding_generator
            
            # PGVector 연결 설정
            connection_string = settings.database_url
            
            # PGVector 벡터 저장소 생성
            vector_store = PGVector(
                collection_name=self.collection_name,
                connection_string=connection_string,
                embedding_function=embedding_function,
            )
            
            logger.info(f"PGVector store initialized: {self.collection_name}")
            return vector_store
        
        except Exception as e:
            logger.error(f"Failed to initialize PGVector store: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """
        문서(코드 청크)를 벡터 DB에 추가
        
        Args:
            documents: LangChain Document 객체 리스트
            batch_size: 배치 크기 (대량 삽입 시 메모리 최적화)
        
        Returns:
            List[str]: 추가된 문서의 ID 리스트
        
        Example:
            >>> rag = RAGPipeline()
            >>> docs = [Document(page_content="def login()...", metadata={"file": "auth.py"})]
            >>> ids = rag.add_documents(docs)
        """
        try:
            all_ids = []
            
            # 배치 단위로 처리 (메모리 최적화)
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # 벡터 DB에 추가
                ids = self.vector_store.add_documents(batch)
                all_ids.extend(ids)
                
                logger.info(
                    f"Added batch {i // batch_size + 1}: "
                    f"{len(batch)} documents (total: {len(all_ids)})"
                )
            
            logger.info(f"Successfully added {len(all_ids)} documents to vector store")
            return all_ids
        
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
            score_threshold: 최소 유사도 점수 (0.0~1.0)
        
        Returns:
            List[Document]: 유사한 문서 리스트
        
        Example:
            >>> rag = RAGPipeline()
            >>> results = rag.search_similar("로그인 기능", top_k=5)
            >>> for doc in results:
            ...     print(doc.page_content)
        """
        try:
            logger.debug(f"Searching for: '{query}' (top_k={top_k})")
            
            if score_threshold:
                # 점수 기반 검색
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=top_k
                )
                # 임계값 필터링
                filtered_results = [
                    doc for doc, score in results if score >= score_threshold
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
            List[tuple[Document, float]]: (문서, 유사도 점수) 튜플 리스트
        
        Example:
            >>> rag = RAGPipeline()
            >>> results = rag.search_with_scores("로그인", top_k=3)
            >>> for doc, score in results:
            ...     print(f"Score: {score:.2f} - {doc.page_content[:50]}")
        """
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
        max_chars: int = 4000
    ) -> str:
        """
        RAG용 컨텍스트 검색 (AI 입력용)
        
        요구사항과 관련된 코드를 검색하여 문자열로 반환합니다.
        
        Args:
            requirement: 요구사항 설명
            top_k: 검색할 청크 개수
            max_chars: 최대 문자 수 (토큰 제한)
        
        Returns:
            str: 검색된 코드 컨텍스트 (AI 입력용)
        
        Example:
            >>> rag = RAGPipeline()
            >>> context = rag.retrieve_context("사용자 인증 기능", top_k=5)
            >>> # AI에게 전달
            >>> analyzer(requirement="...", code_context=context)
        """
        try:
            # 유사 코드 검색
            docs = self.search_similar(query=requirement, top_k=top_k)
            
            # 컨텍스트 문자열 생성
            context_parts = []
            total_chars = 0
            
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
    
    def delete_collection(self) -> None:
        """
        현재 컬렉션 삭제 (테스트/초기화용)
        
        Warning:
            모든 벡터 데이터가 삭제-유의 바람
        """
        try:
            # PGVector 컬렉션 삭제
            self.vector_store.delete_collection()
            logger.warning(f"Collection '{self.collection_name}' deleted")
        
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        벡터 저장소 통계 정보
        
        Returns:
            Dict: 통계 정보
                - collection_name: str
                - embedding_model: str
                - document_count: int (추정)
        """
        try:
            # 샘플 검색으로 저장소 상태 확인
            test_results = self.vector_store.similarity_search(
                query="test",
                k=1
            )
            
            stats = {
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "is_initialized": True,
                "has_documents": len(test_results) > 0
            }
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "is_initialized": False,
                "error": str(e)
            }


# 사용 예시
if __name__ == "__main__":
    import logging
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # RAG 파이프라인 초기화
    rag = RAGPipeline(collection_name="test_collection")
    
    # 샘플 문서 생성
    sample_docs = [
        Document(
            page_content="""
def login(username, password):
    '''사용자 로그인 함수'''
    if username == "admin" and password == "1234":
        return True
    return False
""",
            metadata={"source": "auth.py", "language": "python"}
        ),
        Document(
            page_content="""
def logout(user_id):
    '''사용자 로그아웃 함수'''
    session.delete(user_id)
    return True
""",
            metadata={"source": "auth.py", "language": "python"}
        ),
        Document(
            page_content="""
class User:
    '''사용자 모델'''
    def __init__(self, username):
        self.username = username
""",
            metadata={"source": "models.py", "language": "python"}
        ),
    ]
    
    print("\n=== Test 1: Add Documents ===")
    ids = rag.add_documents(sample_docs)
    print(f"Added {len(ids)} documents")
    
    print("\n=== Test 2: Similarity Search ===")
    results = rag.search_similar("로그인 기능", top_k=2)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"File: {doc.metadata.get('source')}")
        print(f"Content: {doc.page_content[:100]}...")
    
    print("\n=== Test 3: Search with Scores ===")
    results_with_scores = rag.search_with_scores("사용자 인증", top_k=2)
    for doc, score in results_with_scores:
        print(f"\nScore: {score:.2f}")
        print(f"Content: {doc.page_content[:80]}...")
    
    print("\n=== Test 4: Retrieve Context ===")
    context = rag.retrieve_context("로그인 기능 테스트", top_k=2, max_chars=500)
    print(f"Context length: {len(context)} chars")
    print(context[:300] + "...")
    
    print("\n=== Test 5: Get Stats ===")
    stats = rag.get_stats()
    print(f"Stats: {stats}")
    
    # 테스트 컬렉션 삭제
    # rag.delete_collection()
    # print("\nTest collection deleted")