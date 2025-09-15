"""
Base tools for Generic LATS framework.
"""

import logging
from typing import Any, Optional
try:
    from langchain.tools import tool, BaseTool
except ImportError:
    try:
        from langchain_core.tools import tool, BaseTool
    except ImportError:
        # Fallback for basic testing
        def tool(func):
            func._is_langchain_tool = True
            return func
        
        class BaseTool:
            name = "base_tool"
            description = "Base tool"
            
            def _run(self, *args, **kwargs):
                return "Tool result"
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class FinishTool(BaseTool):
    """Tool for submitting final answers and completing episodes."""
    
    name: str = "finish"
    description: str = "Submit the final answer and complete the task. Usage: Finish[answer]"
    
    def _run(self, answer: str, run_manager: Optional[Any] = None) -> str:
        """
        Execute the finish action.
        
        Args:
            answer: The final answer to submit
            run_manager: Optional run manager
            
        Returns:
            Completion message
        """
        logger.info(f"Episode completed with answer: {answer}")
        return f"Episode finished with answer: {answer}"
    
    async def _arun(self, answer: str, run_manager: Optional[Any] = None) -> str:
        """Async version of _run."""
        return self._run(answer, run_manager)


class ThinkTool(BaseTool):
    """Tool for internal reasoning and thought processes."""
    
    name: str = "think"
    description: str = "Express internal thoughts and reasoning. Usage: Think[thought]"
    
    def _run(self, thought: str, run_manager: Optional[Any] = None) -> str:
        """
        Execute the think action.
        
        Args:
            thought: The thought or reasoning to express
            run_manager: Optional run manager
            
        Returns:
            Acknowledgment message
        """
        logger.debug(f"Agent thought: {thought}")
        return "Nice thought."
    
    async def _arun(self, thought: str, run_manager: Optional[Any] = None) -> str:
        """Async version of _run."""
        return self._run(thought, run_manager)


@tool
def simple_calculator(expression: str) -> str:
    """
    Perform basic mathematical calculations.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
        
    Returns:
        Result of the calculation as a string
    """
    try:
        # Basic safety check - only allow certain characters
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        # Evaluate the expression safely
        result = eval(expression)
        return str(result)
        
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def text_search(text: str, query: str) -> str:
    """
    Search for a query string within a text.
    
    Args:
        text: Text to search within
        query: Query string to search for
        
    Returns:
        Found occurrences or "Not found" message
    """
    if not text or not query:
        return "Error: Both text and query must be provided"
    
    query_lower = query.lower()
    text_lower = text.lower()
    
    if query_lower in text_lower:
        # Find all sentences containing the query
        sentences = text.split('.')
        matching_sentences = []
        
        for sentence in sentences:
            if query_lower in sentence.lower():
                matching_sentences.append(sentence.strip())
        
        if matching_sentences:
            return f"Found {len(matching_sentences)} matches:\n" + "\n".join(matching_sentences[:3])
        else:
            return f"Query '{query}' found in text but no complete sentences extracted"
    else:
        return f"Query '{query}' not found in text"


class CustomToolBase(BaseTool):
    """
    Base class for creating custom tools with common functionality.
    
    Subclass this to create domain-specific tools with consistent behavior.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.usage_count = 0
        self.last_result = None
    
    def _run(self, *args, **kwargs) -> str:
        """Base implementation that tracks usage."""
        self.usage_count += 1
        result = self.execute(*args, **kwargs)
        self.last_result = result
        return result
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async version."""
        return self._run(*args, **kwargs)
    
    def execute(self, *args, **kwargs) -> str:
        """
        Override this method to implement tool functionality.
        
        Returns:
            Tool execution result
        """
        raise NotImplementedError("Subclasses must implement execute() method")
    
    def get_usage_stats(self) -> dict:
        """Get usage statistics for this tool."""
        return {
            'usage_count': self.usage_count,
            'last_result': self.last_result
        }


class StateTrackingTool(CustomToolBase):
    """
    Tool that maintains state across invocations.
    
    Useful for tools that need to remember previous operations or results.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = {}
        self.history = []
    
    def execute(self, action: str, *args, **kwargs) -> str:
        """
        Execute an action with state tracking.
        
        Args:
            action: Action to perform
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Result of the action
        """
        # Store the action in history
        self.history.append({
            'action': action,
            'args': args,
            'kwargs': kwargs,
            'timestamp': __import__('time').time()
        })
        
        # Implement action logic here
        return f"Executed action: {action} with args: {args}"
    
    def get_state(self) -> dict:
        """Get current state."""
        return {
            'state': self.state,
            'history_length': len(self.history)
        }
    
    def reset_state(self) -> None:
        """Reset tool state."""
        self.state.clear()
        self.history.clear()


# Factory function for creating simple tools from functions
def create_tool_from_function(func, name: Optional[str] = None, 
                            description: Optional[str] = None) -> BaseTool:
    """
    Create a LangChain tool from a Python function.
    
    Args:
        func: Python function to convert to tool
        name: Optional name for the tool
        description: Optional description for the tool
        
    Returns:
        BaseTool instance
    """
    tool_name = name or func.__name__
    tool_description = description or (func.__doc__ or f"Tool: {tool_name}")
    
    class FunctionTool(BaseTool):
        name: str = tool_name
        description: str = tool_description
        
        def _run(self, *args, **kwargs) -> str:
            try:
                result = func(*args, **kwargs)
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        async def _arun(self, *args, **kwargs) -> str:
            return self._run(*args, **kwargs)
    
    return FunctionTool()


# Predefined tool instances for common use
DEFAULT_TOOLS = {
    'finish': FinishTool(),
    'think': ThinkTool(),
    'calculator': simple_calculator,
    'text_search': text_search,
}