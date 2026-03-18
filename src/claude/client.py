"""
AWS Bedrock Claude API 클라이언트

AWS Bedrock을 통해 Claude 모델과 통신합니다.
"""

from typing import List, Dict, Optional, Any
import boto3
import json
import logging
from config.config import settings

logger = logging.getLogger(__name__)

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
        """
        self.model = model
        self.region = region
        
        try:
            # ⭐ Throttling 방지를 위해 max_attempts를 늘리고 설정을 강화합니다.
            self.client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region,
                config=boto3.session.Config(
                    connect_timeout=timeout,
                    read_timeout=timeout,
                    retries={
                        'max_attempts': 10,  # 재시도 횟수 상향
                        'mode': 'adaptive'   # 속도 조절 모드 활성화
                    }
                )
            )
            logger.info(f"BedrockClient initialized: model={model}, region={region}")
            
            # 🛑 [DISABLED] 연결 테스트는 서버 기동 시 Throttling을 유발하므로 주석 처리합니다.
            # self._test_connection()
            
        except Exception as e:
            error_msg = f"Failed to initialize Bedrock client: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
    
    def _test_connection(self) -> None:
        """Bedrock API 연결 테스트 (현재 비활성화됨)"""
        try:
            test_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hi"}]
            }
            self.client.invoke_model(modelId=self.model, body=json.dumps(test_body))
            logger.info("Bedrock API connection test successful")
        except Exception as e:
            logger.error(f"Failed to connect to Bedrock API: {e}")
            raise ConnectionError(f"Failed to connect to Bedrock API: {e}") from e
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.5, # 💡 기본값을 약간 낮춰서 더 정확한 응답 유도
        max_tokens: int = 4000,    # 💡 충분한 코드 생성을 위해 상향
        system: Optional[str] = None,
        **kwargs
    ) -> str:
        """채팅 완성 요청"""
        try:
            api_messages = [msg for msg in messages if msg["role"] != "system"]
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": api_messages
            }
            
            if system:
                body["system"] = system
            
            if kwargs:
                body.update(kwargs)
            
            response = self.client.invoke_model(
                modelId=self.model,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']
            
            return content
        
        except Exception as e:
            logger.error(f"Bedrock chat failed: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """단일 프롬프트로 텍스트 생성"""
        return self.chat(messages=[{"role": "user", "content": prompt}], **kwargs)

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model": self.model,
            "type": "Claude via AWS Bedrock",
            "region": self.region,
            "status": "initialized"
        }