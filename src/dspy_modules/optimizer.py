"""
DSPy 프롬프트 최적화
참고:
- 실제 최적화를 수행하려면 학습 데이터(라벨링된 예시)가 필요합니다
- 현재는 구조만 제공하며, 추후 데이터가 축적되면 활성화할 수 있습니다
"""

import dspy
from typing import List, Dict, Any, Tuple
import logging

from .signature import ExecutabilitySignature
from .dspy_modules import ExecutabilityAnalyzer

logger = logging.getLogger(__name__)


class OptimizerConfig:
    """
    최적화 설정
    
    Attributes:
        num_trials: 최적화 시도 횟수
        metric_threshold: 성공 기준 점수
        max_bootstrapped_demos: 최대 예시 개수
    """
    
    def __init__(
        self,
        num_trials: int = 10,
        metric_threshold: float = 0.8,
        max_bootstrapped_demos: int = 4
    ):
        self.num_trials = num_trials
        self.metric_threshold = metric_threshold
        self.max_bootstrapped_demos = max_bootstrapped_demos


class ExecutabilityOptimizer:
    """
    ExecutabilityAnalyzer 최적화기
    
    학습 데이터를 기반으로 프롬프트를 자동 최적화합니다.
    
    사용 시나리오:
    1. 수동으로 라벨링한 데이터 축적 (최소 20개 이상 권장)
    2. Optimizer로 프롬프트 최적화
    3. 최적화된 모델 저장 및 사용
    
    Example:
        >>> # 1. 학습 데이터 준비
        >>> train_data = [
        ...     {
        ...         "requirement": "로그인 기능",
        ...         "code_context": "def login()...",
        ...         "is_executable": True,
        ...         "reasoning": "login() 함수가 구현됨"
        ...     },
        ...     ...
        ... ]
        >>> 
        >>> # 2. 최적화 수행
        >>> optimizer = ExecutabilityOptimizer()
        >>> optimized_model = optimizer.optimize(train_data, val_data)
        >>> 
        >>> # 3. 최적화된 모델 사용
        >>> result = optimized_model(requirement="...", code_context="...")
    """
    
    def __init__(self, config: OptimizerConfig = None):
        """
        Optimizer 초기화
        
        Args:
            config: 최적화 설정 (기본값 사용 가능)
        """
        self.config = config or OptimizerConfig()
        logger.info(f"ExecutabilityOptimizer initialized: {self.config.__dict__}")
    
    def prepare_dataset(
        self,
        raw_data: List[Dict[str, Any]]
    ) -> List[dspy.Example]:
        """
        원시 데이터를 DSPy Example로 변환
        
        Args:
            raw_data: 라벨링된 데이터
                [
                    {
                        "requirement": str,
                        "code_context": str,
                        "is_executable": bool,
                        "reasoning": str,
                        "confidence_score": int
                    },
                    ...
                ]
        
        Returns:
            List[dspy.Example]: DSPy 학습 데이터
        """
        examples = []
        
        for item in raw_data:
            # bool → str 변환
            is_executable = "true" if item["is_executable"] else "false"
            confidence_score = str(item.get("confidence_score", 50))
            
            example = dspy.Example(
                requirement=item["requirement"],
                code_context=item["code_context"],
                is_executable=is_executable,
                reasoning=item.get("reasoning", ""),
                confidence_score=confidence_score
            ).with_inputs("requirement", "code_context")
            
            examples.append(example)
        
        logger.info(f"Prepared {len(examples)} examples for training")
        return examples
    
    def accuracy_metric(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace=None
    ) -> float:
        """
        정확도 평가 메트릭
        
        Args:
            example: 실제 정답
            prediction: 모델 예측
            trace: 추적 정보 (디버깅용)
        
        Returns:
            float: 정확도 점수 (0.0~1.0)
        """
        # is_executable 일치 여부
        is_correct = (
            example.is_executable.lower() == prediction.is_executable.lower()
        )
        
        # 기본 점수
        score = 1.0 if is_correct else 0.0
        
        # reasoning이 있으면 보너스 점수
        if is_correct and len(prediction.reasoning) > 20:
            score += 0.2  # 최대 1.2점
        
        return min(score, 1.0)
    
    def optimize(
        self,
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]] = None,
        optimizer_type: str = "BootstrapFewShot"
    ) -> ExecutabilityAnalyzer:
        """
        프롬프트 최적화 수행
        
        Args:
            train_data: 학습 데이터
            val_data: 검증 데이터 (없으면 train_data의 20% 사용)
            optimizer_type: 최적화 알고리즘
                - "BootstrapFewShot": Few-shot 학습
                - "MIPRO": Multi-stage Instruction Optimization
        
        Returns:
            ExecutabilityAnalyzer: 최적화된 모델
        
        Raises:
            ValueError: 학습 데이터가 부족한 경우
        """
        if len(train_data) < 10:
            raise ValueError(
                f"최소 10개 이상의 학습 데이터가 필요합니다. "
                f"현재: {len(train_data)}개"
            )
        
        logger.info(f"Starting optimization with {len(train_data)} training examples")
        
        # 1. 데이터 준비
        train_examples = self.prepare_dataset(train_data)
        
        if val_data:
            val_examples = self.prepare_dataset(val_data)
        else:
            # 학습 데이터의 20%를 검증용으로 사용
            split_idx = int(len(train_examples) * 0.8)
            train_examples, val_examples = (
                train_examples[:split_idx],
                train_examples[split_idx:]
            )
        
        logger.info(
            f"Split: {len(train_examples)} train, {len(val_examples)} validation"
        )
        
        # 2. 최적화기 선택
        if optimizer_type == "BootstrapFewShot":
            optimizer = dspy.BootstrapFewShot(
                metric=self.accuracy_metric,
                max_bootstrapped_demos=self.config.max_bootstrapped_demos,
                max_labeled_demos=self.config.max_bootstrapped_demos
            )
        elif optimizer_type == "MIPRO":
            optimizer = dspy.MIPRO(
                metric=self.accuracy_metric,
                num_trials=self.config.num_trials,
                prompt_model=dspy.settings.lm
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # 3. 최적화 수행
        logger.info(f"Compiling with {optimizer_type}...")
        
        base_model = ExecutabilityAnalyzer()
        optimized_model = optimizer.compile(
            base_model,
            trainset=train_examples,
            valset=val_examples
        )
        
        # 4. 검증
        logger.info("Evaluating optimized model...")
        evaluation_score = self._evaluate(optimized_model, val_examples)
        
        logger.info(
            f"Optimization complete! "
            f"Validation accuracy: {evaluation_score:.2%}"
        )
        
        return optimized_model
    
    def _evaluate(
        self,
        model: ExecutabilityAnalyzer,
        examples: List[dspy.Example]
    ) -> float:
        """
        모델 평가
        
        Args:
            model: 평가할 모델
            examples: 평가 데이터
        
        Returns:
            float: 평균 정확도
        """
        total_score = 0.0
        
        for example in examples:
            prediction = model(
                requirement=example.requirement,
                code_context=example.code_context
            )
            score = self.accuracy_metric(example, prediction)
            total_score += score
        
        return total_score / len(examples)
    
    def save_optimized_model(
        self,
        model: ExecutabilityAnalyzer,
        filepath: str = "optimized_analyzer.json"
    ) -> None:
        """
        최적화된 모델 저장
        
        Args:
            model: 저장할 모델
            filepath: 저장 경로
        """
        try:
            model.save(filepath)
            logger.info(f"Optimized model saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise
    
    def load_optimized_model(
        self,
        filepath: str = "optimized_analyzer.json"
    ) -> ExecutabilityAnalyzer:
        """
        저장된 모델 로드
        
        Args:
            filepath: 모델 파일 경로
        
        Returns:
            ExecutabilityAnalyzer: 로드된 모델
        """
        try:
            model = ExecutabilityAnalyzer()
            model.load(filepath)
            logger.info(f"Optimized model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise


# 사용 예시
if __name__ == "__main__":
    import logging
    from .dspy_modules import configure_dspy
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("\n=== DSPy Optimizer Demo ===")
    print("참고: 실제 최적화를 수행하려면 라벨링된 학습 데이터가 필요합니다.")
    
    # DSPy 설정
    configure_dspy(
        base_url="http://localhost:11434",
        model="llama3.1:8b"
    )
    
    # 샘플 학습 데이터 (실제로는 최소 20개 이상 필요)
    sample_train_data = [
        {
            "requirement": "사용자 로그인 기능",
            "code_context": "def login(username, password): return True",
            "is_executable": True,
            "reasoning": "login() 함수가 구현되어 있음",
            "confidence_score": 90
        },
        {
            "requirement": "데이터베이스 연결",
            "code_context": "def connect(): pass",
            "is_executable": False,
            "reasoning": "connect() 함수가 비어있음",
            "confidence_score": 30
        },
        # ... 더 많은 데이터 필요
    ]
    
    print(f"\n학습 데이터 개수: {len(sample_train_data)}")
    print("실제 최적화를 위해서는 최소 10개 이상의 데이터가 필요합니다.")
    
    # Optimizer 초기화
    optimizer = ExecutabilityOptimizer()
    
    # 데이터 준비 테스트
    examples = optimizer.prepare_dataset(sample_train_data)
    print(f"\n준비된 Example 개수: {len(examples)}")
    print(f"첫 번째 Example: {examples[0]}")
    
    print("\n참고: 충분한 데이터가 있다면 optimizer.optimize()를 호출하세요.")