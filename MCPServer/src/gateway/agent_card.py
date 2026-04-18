from pydantic import BaseModel
from typing import List, Optional

class AgentCapability(BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

class AgentCard(BaseModel):
    agent_id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    version: str = "1.0.0"
