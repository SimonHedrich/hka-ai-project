from pydantic import BaseModel, Field
from typing import Literal, List, Any, Union, Optional, Annotated
from configs.optimizer_schema import BaseOptimizerConfig

class BaseSchedulerConfig(BaseModel):
    optimizer: Optional[BaseOptimizerConfig] = None
    model_config = {
        "arbitrary_types_allowed": True
    }

class ExponentialLRConfig(BaseSchedulerConfig):
    target: Literal["torch.optim.lr_scheduler.ExponentialLR"] = Field(...)
    gamma: float

class LinearLRConfig(BaseSchedulerConfig):
    target: Literal["torch.optim.lr_scheduler.LinearLR"] = Field(...)
    start_factor: float
    total_iters: int

class MultiStepLRConfig(BaseSchedulerConfig):
    target: Literal["torch.optim.lr_scheduler.MultiStepLR"] = Field(...)
    milestones: List[int]
    gamma: float = 0.1

class OneCycleLRConfig(BaseSchedulerConfig):
    target: Literal["torch.optim.lr_scheduler.OneCycleLR"] = Field(...)
    max_lr: float
    steps_per_epoch: int
    epochs: int
    pct_start: float
    anneal_strategy: str = 'cos'
    div_factor: float
    final_div_factor: float
    total_steps: int

class PolynomialLRConfig(BaseSchedulerConfig):
    target: Literal["torch.optim.lr_scheduler.PolynomialLR"] = Field(...)
    total_iters: int
    power: int
    last_epoch: int
    optimizer: Any = None

class ReduceLROnPlateauConfig(BaseSchedulerConfig):
    target: Literal["torch.optim.lr_scheduler.ReduceLROnPlateau"] = Field(...)
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10


# Define new Schedulers here


SchedulerConfig = Annotated[
    Union[
    ExponentialLRConfig,
    LinearLRConfig,
    MultiStepLRConfig,
    OneCycleLRConfig,
    PolynomialLRConfig,
    ReduceLROnPlateauConfig
    # Add new defined Schedulers here
    ],
    Field(discriminator="target")
]
