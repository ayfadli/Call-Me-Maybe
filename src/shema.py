from pydantic import BaseModel
from typing import Dict, Any

class ParameterDef(BaseModel):
    type: str

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef

class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]

