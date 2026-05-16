import dspy
import litellm
from typing import Dict, Any
from config.config import settings
import logging
logger = logging.getLogger(__name__)

# response_format 강제 비활성화
litellm.add_function_to_prompt = True

def configure_bedrock_dspy(
    model=settings.bedrock_model,
    region=settings.aws_region,
    **kwargs
) -> None:
    try:
        import boto3
        boto3.setup_default_session(region_name=region)

        lm = dspy.LM(
            f"bedrock/{settings.bedrock_model}",
            temperature=0.1,
            max_tokens=16000,
            cache=False,
            aws_region_name=region,
        )

        dspy.settings.configure(lm=lm)

        logger.info(
            f"DSPy configured with Bedrock: "
            f"model={model}, region={region}"
        )

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
