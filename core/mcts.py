from .wfc_env import WFCWrapper
from copy import deepcopy
from pydantic import BaseModel, Field
from

class Action(BaseModel):
    action_logits: list[float] = Field(default_factory=list, description="Logits for the action taken")
    visits: int = Field(default=0, description="Number of times this action has been taken")
    total_reward: float | None = Field(default=None, description="Total reward accumulated from this action")

class Node:
    def __init__(self, env: WFCWrapper, parent=None):
        self.env = deepcopy(env)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_reward: float | None = None
        
        
