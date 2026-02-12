"""
AWS Bedrock Claude API 클라이언트

AWS Bedrock을 통해 Claude 모델과 통신합니다.
"""

from typing import List, Dict, Optional, Any
import boto3
import json
import logging
from config.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class BedrockClient:
    """
    AWS Bedrock Claude API 클라이언트
    """
    
    def __init__(
        self,
        model: str = settings.bedrock_model, 
        region: str = settings.aws_region,
        timeout: int = 60
    ):
        """
        BedrockClient 초기화
        
        Args:
            model: Claude 모델 ID (Bedrock 형식)
            region: AWS 리전
            timeout: 요청 타임아웃 (초)
        
        Note:
            AWS 인증은 IAM Role 또는 환경변수(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)를 사용합니다.
        """
        self.model = model
        self.region = region
        
        try:
            session = boto3.Session(region_name=self.region)
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region,
                config=boto3.session.Config(
                    connect_timeout=timeout,
                    read_timeout=timeout,
                    retries={'max_attempts': 3}
                )
            )
            logger.info(f"BedrockClient initialized: model={model}, region={region}")
            
            # 연결 테스트
            self._test_connection()
            
        except Exception as e:
            error_msg = f"Failed to initialize Bedrock client: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
    
    def _test_connection(self) -> None:
        """Bedrock API 연결 테스트"""
        try:
            # 간단한 테스트 요청
            test_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [
                    {"role": "user", "content": "Hi"}
                ]
            }
            
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps(test_body)
            )
            
            logger.info("Bedrock API connection test successful")
            
        except Exception as e:
            error_msg = f"Failed to connect to Bedrock API: {e}"
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
            logger.debug(f"Bedrock chat request: {len(messages)} messages")
            
            # Bedrock API 형식으로 변환
            api_messages = []
            for msg in messages:
                if msg["role"] != "system":  # system은 별도 처리
                    api_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Request body 구성
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": api_messages
            }
            
            # System 프롬프트 추가
            if system:
                body["system"] = system
            
            # 추가 파라미터
            if kwargs:
                body.update(kwargs)
            
            # API 호출
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps(body)
            )
            
            # 응답 파싱
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            logger.debug(f"Bedrock response: {len(content)} characters")
            
            return content
        
        except Exception as e:
            logger.error(f"Bedrock chat failed: {e}")
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
            "type": "Claude via AWS Bedrock",
            "region": self.region,
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
        # IAM Role 또는 환경변수로 인증
        client = BedrockClient(
            region="us-east-1"  # 필요시 변경
        )
        
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
        print("\nMake sure:")
        print("1. AWS credentials are configured (IAM Role or environment variables)")
        print("2. Bedrock is available in your region")
        print("3. You have access to Claude models in Bedrock")