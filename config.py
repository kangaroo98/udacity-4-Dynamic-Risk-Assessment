'''
Exception Handling - Definition

Author: Oliver
Date: 2022 - MArch
'''
from pydantic import BaseModel
from typing import Literal, Union 
from shtab import Optional

from datetime import datetime

# pydantiy model
METRIC = Literal['F1', 'MAE', 'SSE']
MODE = Literal['train','test']

class Score(BaseModel):
    version: int = None 
    mode: MODE
    timestamp: datetime
    metric: METRIC
    score: float
    
    def __getitem__(self, item):
        return getattr(self, item)

# define user-defined exceptions
class AppError(Exception):
    """Base class for other exceptions"""
    pass

class UnsupportedFileType(AppError):
    """Base class for other exceptions"""
    pass

class FileInvalid(AppError):
    """Base class for other exceptions"""
    pass

class DeploymentFailed(AppError):
    """Base class for other exceptions"""
    pass