"""
Code Chunking using LangChain Text Splitters
"""
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)


class CodeChunker:
    """코드를 의미 있는 청크로 분할하는 클래스"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Args:
            chunk_size: 청크의 최대 크기 (문자 수)
            chunk_overlap: 청크 간 오버랩 (문자 수)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_documents(
        self,
        documents: List[Document],
        language: str = "python"
    ) -> List[Document]:
        """
        문서를 청크로 분할
        
        Args:
            documents: LangChain Document 객체 리스트
            language: 프로그래밍 언어 (python, javascript, java 등)
        
        Returns:
            List[Document]: 분할된 청크들
        """
        try:
            # 언어별 TextSplitter 생성
            text_splitter = self._create_splitter(language)
            
            # 문서 분할
            chunks = text_splitter.split_documents(documents)
            
            logger.info(
                f"Split {len(documents)} documents into {len(chunks)} chunks"
            )
            
            return chunks
        
        except Exception as e:
            logger.error(f"Failed to split documents: {e}")
            raise
    
    def _create_splitter(self, language: str) -> RecursiveCharacterTextSplitter:
        """
        언어별 TextSplitter 생성
        
        Args:
            language: 프로그래밍 언어
        
        Returns:
            RecursiveCharacterTextSplitter
        """
        # LangChain이 지원하는 언어 매핑
        language_map = {
            "python": Language.PYTHON,
            "javascript": Language.JS,
            "java": Language.JAVA,
        }
        
        lang = language_map.get(language.lower())
        
        if lang:
            # 언어별 최적화된 분할
            return RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        else:
            # 범용 분할
            logger.warning(
                f"Language '{language}' not supported, using generic splitter"
            )
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
    
    def split_by_file_extension(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        파일 확장자를 기반으로 자동으로 언어를 감지하여 분할
        
        Args:
            documents: LangChain Document 객체 리스트
        
        Returns:
            List[Document]: 분할된 청크들
        """
        extension_to_language = {
            ".py": "python",
            ".js": "javascript",
            ".java": "java",
        }
        
        all_chunks = []
        
        for doc in documents:
            # 파일 경로에서 확장자 추출
            file_path = doc.metadata.get("source", "")
            extension = None
            
            for ext in extension_to_language.keys():
                if file_path.endswith(ext):
                    extension = ext
                    break
            
            # 언어 결정
            language = extension_to_language.get(extension, "python")
            
            # 분할
            text_splitter = self._create_splitter(language)
            chunks = text_splitter.split_documents([doc])
            
            all_chunks.extend(chunks)
        
        logger.info(
            f"Split {len(documents)} documents into {len(all_chunks)} chunks "
            f"using file extension detection"
        )
        
        return all_chunks


# 사용 예시
if __name__ == "__main__":
    from langchain.schema import Document
    
    # 샘플 문서
    sample_code = """
    def calculate_sum(numbers):
        \"\"\"숫자 리스트의 합계를 계산합니다.\"\"\"
        total = 0
        for num in numbers:
            total += num
        return total
    
    def calculate_average(numbers):
        \"\"\"숫자 리스트의 평균을 계산합니다.\"\"\"
        if not numbers:
            return 0
        return calculate_sum(numbers) / len(numbers)
    """
    
    docs = [Document(page_content=sample_code, metadata={"source": "test.py"})]
    
    chunker = CodeChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.split_documents(docs, language="python")
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk.page_content)