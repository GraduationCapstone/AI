"""
Claude API 클라이언트

Anthropic Claude API와 통신합니다.
"""

from typing import List, Dict, Optional, Any
from anthropic import Anthropic
import logging

logger = logging.getLogger(__name__)


class ClaudeClient:
    """
    Anthropic Claude API 클라이언트
    
    Attributes:
        client: Anthropic 클라이언트
        model: Claude 모델 이름
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        timeout: int = 60
    ):
        """
        ClaudeClient 초기화
        
        Args:
            api_key: Anthropic API 키
            model: Claude 모델 이름
            timeout: 요청 타임아웃 (초)
        """
        if not api_key:
            raise ValueError("Anthropic API key is required")
        
        self.model = model
        self.client = Anthropic(
            api_key=api_key,
            timeout=timeout
        )
        
        logger.info(f"ClaudeClient initialized: model={model}")
        
        # 연결 테스트
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Claude API 연결 테스트"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            logger.info("Claude API connection test successful")
        except Exception as e:
            error_msg = f"Failed to connect to Claude API: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        채팅 완성 요청
        
        Args:
            messages: 메시지 리스트 (role, content)
            temperature: 생성 온도
            max_tokens: 최대 토큰 수
            system: 시스템 프롬프트
        
        Returns:
            str: 생성된 응답
        """
        try:
            logger.debug(f"Claude chat request: {len(messages)} messages")
            
            # Anthropic API 형식으로 변환
            api_messages = []
            for msg in messages:
                if msg["role"] != "system":  # system은 별도 처리
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # API 호출
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=api_messages,
                **kwargs
            )
            
            content = response.content[0].text
            logger.debug(f"Claude response: {len(content)} characters")
            
            return content
        
        except Exception as e:
            logger.error(f"Claude chat failed: {e}")
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
        messages = [{"role": "user", "content": prompt}]
        
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system=system_message,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model": self.model,
            "type": "Claude API (Anthropic)",
            "status": "connected"
        }


# 사용 예시
if __name__ == "__main__":
    import logging
    import os
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
    else:
        try:
            client = ClaudeClient(api_key=api_key)
            
            print("\n=== Test 1: Simple Generation ===")
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