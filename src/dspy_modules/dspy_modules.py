"""
DSPy 설정 모듈

AWS Bedrock 기반 DSPy 설정 함수를 제공합니다.
"""

import dspy
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def configure_bedrock_dspy(
    model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    region: str = "us-east-1",
    **kwargs
) -> None:
    """
    DSPy 설정 (AWS Bedrock 연동)
    
    애플리케이션 시작 시 한 번만 호출하면 됩니다.
    
    Args:
        model: Claude 모델 ID
        region: AWS 리전
        **kwargs: BedrockClient에 전달할 추가 인자
    
    Example:
        >>> from src.dspy_modules import configure_bedrock_dspy
        >>> configure_bedrock_dspy(region="us-east-1")
    """
    try:
        from .bedrock_lm import BedrockLM
        
        # Bedrock LM 초기화
        bedrock_lm = BedrockLM(
            model=model,
            region=region,
            **kwargs
        )
        
        # DSPy 설정
        dspy.settings.configure(lm=bedrock_lm)
        
        logger.info(
            f"DSPy configured with Bedrock: "
            f"model={model}, region={region}"
        )
        
        # 연결 테스트
        test_lm = dspy.settings.lm
        logger.info(f"DSPy LM configured: {test_lm}")
    
    except Exception as e:
        logger.error(f"Failed to configure DSPy with Bedrock: {e}")
        raise ConnectionError(
            f"Cannot connect to Bedrock in {region}. "
            f"Please ensure AWS credentials are configured and "
            f"you have access to Claude models in Bedrock."
        ) from e


def get_current_lm_info() -> Dict[str, Any]:
    """
    현재 설정된 LM 정보 반환
    
    Returns:
        Dict: LM 정보
    """
    try:
        lm = dspy.settings.lm
        return {
            "configured": True,
            "model": str(lm),
            "type": type(lm).__name__
        }
    except AttributeError:
        return {
            "configured": False,
            "error": "DSPy LM not configured. Call configure_bedrock_dspy() first."
        }


# 사용 예시
if __name__ == "__main__":
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== Test 1: Configure Bedrock DSPy ===")
    try:
        configure_bedrock_dspy(region="us-east-1")
        print("DSPy configured successfully")
        
        print("\n=== Test 2: Check LM Info ===")
        lm_info = get_current_lm_info()
        print(f"LM Info: {lm_info}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure AWS credentials are configured")