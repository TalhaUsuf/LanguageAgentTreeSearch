"""
Generic Language Agent Tree Search (LATS) Framework

A flexible implementation of LATS, ToT, and RAP algorithms for any task domain
with configurable prompts, tools, and LLM backends via LangChain integration.
"""

__version__ = "1.0.0"
__author__ = "Generic LATS Framework"

try:
    from .core.lats_algorithm import LATSSearch
    from .tasks.base_task import BaseTask
    from .tools.tool_registry import ToolRegistry
    from .models.llm_wrapper import LangChainLLM
    from .utils.config_loader import ConfigLoader
except ImportError:
    # For direct execution without package installation
    from core.lats_algorithm import LATSSearch
    from tasks.base_task import BaseTask
    from tools.tool_registry import ToolRegistry
    from models.llm_wrapper import LangChainLLM
    from utils.config_loader import ConfigLoader

__all__ = [
    "LATSSearch",
    "BaseTask", 
    "ToolRegistry",
    "LangChainLLM",
    "ConfigLoader"
]