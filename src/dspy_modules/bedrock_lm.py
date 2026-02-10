"""
DSPy용 Bedrock LM 래퍼

AWS Bedrock Claude를 DSPy Language Model로 사용할 수 있게 래핑합니다.
"""

import dspy
from typing import List, Dict, Any, Optional
import logging

from src.claude import BedrockClient

logger = logging.getLogger(__name__)


class BedrockLM(dspy.LM):
    """
    DSPy용 AWS Bedrock 언어 모델
    
    BedrockClient를 DSPy의 LM 인터페이스로 래핑합니다.
    
    Attributes:
        client: BedrockClient 인스턴스
        model: Claude 모델 ID
        kwargs: 기본 생성 파라미터
    
    Example:
        >>> import dspy
        >>> from src.dspy_modules import BedrockLM
        >>> 
        >>> # DSPy 설정
        >>> bedrock_lm = BedrockLM(region="us-east-1")
        >>> dspy.settings.configure(lm=bedrock_lm)
        >>> 
        >>> # 사용
        >>> generator = dspy.ChainOfThought(signature)
        >>> result = generator(...)
    """
    
    def __init__(
        self,
        model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
        **kwargs
    ):
        """
        BedrockLM 초기화
        
        Args:
            model: Claude 모델 ID
            region: AWS 리전
            **kwargs: BedrockClient에 전달할 추가 인자
        """
        super().__init__(model=model)
        
        self.client = BedrockClient(
            model=model,
            region=region,
            **kwargs
        )
        
        self.model = model
        self.kwargs = {
            "temperature": 0.7,
            "max_tokens": 2000,
            **kwargs
        }
        
        logger.info(f"BedrockLM initialized: model={model}, region={region}")
    
    def basic_request(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """
        DSPy 기본 요청 인터페이스
        
        Args:
            prompt: 프롬프트 텍스트
            **kwargs: 생성 파라미터 오버라이드
        
        Returns:
            str: 생성된 텍스트
        """
        # 파라미터 병합
        params = {**self.kwargs, **kwargs}
        
        try:
            # Bedrock 호출
            response = self.client.generate(
                prompt=prompt,
                **params
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Bedrock request failed: {e}")
            raise
    
    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> List[str]:
        """
        DSPy 호출 인터페이스
        
        Args:
            prompt: 단일 프롬프트 (또는)
            messages: 메시지 리스트
            **kwargs: 생성 파라미터
        
        Returns:
            List[str]: 생성된 응답 리스트 (보통 1개)
        """
        # 파라미터 병합
        params = {**self.kwargs, **kwargs}
        
        try:
            if messages:
                # 메시지 형식
                response = self.client.chat(
                    messages=messages,
                    **params
                )
            elif prompt:
                # 단일 프롬프트
                response = self.client.generate(
                    prompt=prompt,
                    **params
                )
            else:
                raise ValueError("Either prompt or messages must be provided")
            
            # DSPy는 리스트 형태로 반환 기대
            return [response]
        
        except Exception as e:
            logger.error(f"Bedrock call failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return self.client.get_model_info()
    
    def __repr__(self) -> str:
        return f"BedrockLM(model='{self.model}')"


# 사용 예시
if __name__ == "__main__":
    import logging
    import dspy
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        print("\n=== Test 1: Initialize BedrockLM ===")
        bedrock_lm = BedrockLM(region="us-east-1")
        print(f"Model: {bedrock_lm}")
        
        print("\n=== Test 2: Basic Request ===")
        response = bedrock_lm.basic_request(
            "What is Python? Answer in one sentence."
        )
        print(f"Response: {response}")
        
        print("\n=== Test 3: Configure DSPy ===")
        dspy.settings.configure(lm=bedrock_lm)
        print("DSPy configured with BedrockLM")
        
        # 간단한 Signature 테스트
        class QASignature(dspy.Signature):
            """Answer questions briefly."""
            question = dspy.InputField()
            answer = dspy.OutputField()
        
        print("\n=== Test 4: DSPy ChainOfThought ===")
        qa = dspy.ChainOfThought(QASignature)
        result = qa(question="What is 2+2?")
        print(f"Answer: {result.answer}")
        
        print("\n=== Test 5: Model Info ===")
        info = bedrock_lm.get_model_info()
        print(f"Model Info: {info}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure:")
        print("1. AWS credentials are configured")
        print("2. Bedrock is available in your region")
        print("3. You have access to Claude models")