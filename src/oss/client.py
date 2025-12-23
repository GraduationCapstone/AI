"""
GPT OSS (Ollama) API 클라이언트

Ollama를 통해 로컬에서 실행되는 GPT OSS 모델과 통신합니다.
Ollama는 OpenAI 호환 API를 제공하므로 openai 패키지를 사용합니다.
"""

from typing import List, Dict, Optional, Any
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)


class GPTClient:
    """
    Ollama를 통한 GPT OSS 모델 클라이언트
    
    Ollama는 로컬에서 LLM을 실행하는 도구로, OpenAI 호환 API를 제공합니다.
    
    Attributes:
        client: OpenAI 클라이언트 인스턴스
        model: 사용할 GPT 모델 이름
        base_url: Ollama 서버 URL
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "gpt-3.5-turbo",
        timeout: int = 60
    ):
        """
        GPTClient 초기화
        
        Args:
            base_url: Ollama 서버 URL
            model: 사용할 GPT 모델 이름
            timeout: 요청 타임아웃 (초)
        """
        self.base_url = base_url
        self.model = model
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama는 API 키 불필요 (더미 값)
            timeout=timeout
        )
        
        logger.info(f"GPTClient initialized: {base_url}, model={model}")
        
        # 연결 테스트
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Ollama 서버 연결 테스트"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            logger.info("GPT OSS (Ollama) connection test successful")
        except Exception as e:
            error_msg = (
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Please ensure Ollama is running and model '{self.model}' is installed."
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        채팅 완성 요청
        
        Args:
            messages: 메시지 리스트
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
        
        Returns:
            str: 생성된 응답
        """
        try:
            logger.debug(f"GPT chat request: {len(messages)} messages")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            content = response.choices[0].message.content
            logger.debug(f"GPT response: {len(content)} characters")
            
            return content
        
        except Exception as e:
            logger.error(f"GPT chat failed: {e}")
            raise
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        단일 프롬프트로 텍스트 생성
        
        Args:
            prompt: 사용자 프롬프트
            system_message: 시스템 메시지
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
        
        Returns:
            str: 생성된 응답
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "type": "GPT OSS (Ollama)",
            "status": "connected"
        }


# 사용 예시
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        client = GPTClient(
            base_url="http://localhost:11434/v1",
            model="gpt-3.5-turbo"
        )
        
        print("\n=== Test 1: Simple Chat ===")
        response = client.generate(
            "What is Python?",
            system_message="You are a helpful assistant."
        )
        print(f"Response: {response[:200]}...")
        
        print("\n=== Test 2: Model Info ===")
        info = client.get_model_info()
        print(f"Model Info: {info}")
        
    except ConnectionError as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Install Ollama: https://ollama.com/")
        print("2. Run: ollama serve")
        print("3. Pull model: ollama pull gpt-3.5-turbo")