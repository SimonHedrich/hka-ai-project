from pydantic import BaseModel, Field
from typing import Literal, Union, Optional, Annotated, Tuple
from torch.optim.optimizer import ParamsT

class BaseOptimizerConfig(BaseModel):
    params: Optional[ParamsT] = None
    model_config = {
        "arbitrary_types_allowed": True
    }

class SGDOptimizerConfig(BaseOptimizerConfig):
    target: Literal["torch.optim.SGD"] = Field(...)
    lr: float
    momentum: float
    weight_decay: float

class AdamOptimizerConfig(BaseOptimizerConfig):
    target: Literal["torch.optim.Adam"] = Field(...)
    lr: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    
# Define new optimizer here

OptimizerConfig = Annotated[
    Union[
    SGDOptimizerConfig,
    AdamOptimizerConfig
    # Add new defined optimizers here
    ],
    Field(discriminator="target")
]