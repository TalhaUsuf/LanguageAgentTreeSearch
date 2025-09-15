"""
Tool registry for managing and accessing tools in Generic LATS framework.
"""

import logging
from typing import Dict, List, Any, Optional, Type, Callable
from pathlib import Path
import importlib.util
import inspect
from langchain.tools import Tool, BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for managing tools that can be used by the LATS algorithm.
    
    Supports registration of:
    - LangChain Tool objects
    - Python functions decorated with @tool
    - Custom tool classes
    - Dynamic loading from directories
    """
    
    def __init__(self):
        """Initialize empty tool registry."""
        self.tools: Dict[str, Any] = {}
        self.tool_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initialized ToolRegistry")
    
    def register(self, name: str, tool: Any, description: Optional[str] = None, 
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a tool in the registry.
        
        Args:
            name: Unique name for the tool
            tool: Tool object (LangChain Tool, function, or custom tool)
            description: Optional description override
            metadata: Optional metadata dictionary
        """
        if name in self.tools:
            logger.warning(f"Tool '{name}' already registered, overwriting")
        
        self.tools[name] = tool
        
        # Store metadata
        tool_meta = metadata or {}
        if description:
            tool_meta['description'] = description
        elif hasattr(tool, 'description'):
            tool_meta['description'] = tool.description
        elif hasattr(tool, '__doc__') and tool.__doc__:
            tool_meta['description'] = tool.__doc__.strip()
        else:
            tool_meta['description'] = f"Tool: {name}"
        
        self.tool_metadata[name] = tool_meta
        
        logger.info(f"Registered tool: {name}")
    
    def register_function(self, func: Callable, name: Optional[str] = None,
                         description: Optional[str] = None) -> None:
        """
        Register a Python function as a tool.
        
        Args:
            func: Python function to register
            name: Optional name for the tool (defaults to function name)
            description: Optional description (defaults to docstring)
        """
        tool_name = name or func.__name__
        tool_description = description or (func.__doc__.strip() if func.__doc__ else f"Function: {tool_name}")
        
        # Create a LangChain Tool from the function
        langchain_tool = Tool(
            name=tool_name,
            description=tool_description,
            func=func
        )
        
        self.register(tool_name, langchain_tool, tool_description)
    
    def register_class(self, tool_class: Type[BaseTool], name: Optional[str] = None,
                      **kwargs) -> None:
        """
        Register a tool class (subclass of BaseTool).
        
        Args:
            tool_class: Class that inherits from BaseTool
            name: Optional name for the tool
            **kwargs: Arguments to pass to the tool constructor
        """
        tool_name = name or tool_class.__name__.lower().replace('tool', '')
        
        # Instantiate the tool
        tool_instance = tool_class(**kwargs)
        
        self.register(tool_name, tool_instance)
    
    def get_tool(self, name: str) -> Any:
        """
        Get a tool by name.
        
        Args:
            name: Name of the tool
            
        Returns:
            Tool object
            
        Raises:
            KeyError: If tool not found
        """
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        
        return self.tools[name]
    
    def get_tool_names(self) -> List[str]:
        """
        Get list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def get_tools(self) -> List[Any]:
        """
        Get list of all registered tools.
        
        Returns:
            List of tool objects
        """
        return list(self.tools.values())
    
    def get_langchain_tools(self) -> List[BaseTool]:
        """
        Get list of tools compatible with LangChain.
        
        Returns:
            List of LangChain-compatible tools
        """
        langchain_tools = []
        
        for name, tool in self.tools.items():
            if isinstance(tool, BaseTool):
                langchain_tools.append(tool)
            elif callable(tool):
                # Convert function to LangChain Tool
                description = self.tool_metadata.get(name, {}).get('description', f"Tool: {name}")
                langchain_tool = Tool(
                    name=name,
                    description=description,
                    func=tool
                )
                langchain_tools.append(langchain_tool)
        
        return langchain_tools
    
    def get_tool_description(self, name: str) -> str:
        """
        Get description for a specific tool.
        
        Args:
            name: Tool name
            
        Returns:
            Tool description string
        """
        if name not in self.tool_metadata:
            return f"Tool: {name}"
        
        return self.tool_metadata[name].get('description', f"Tool: {name}")
    
    def get_tools_description(self) -> str:
        """
        Get formatted description of all tools.
        
        Returns:
            Formatted string describing all tools
        """
        if not self.tools:
            return "No tools available."
        
        descriptions = []
        for name in self.tools.keys():
            desc = self.get_tool_description(name)
            descriptions.append(f"- {name}: {desc}")
        
        return "\n".join(descriptions)
    
    def unregister(self, name: str) -> None:
        """
        Remove a tool from the registry.
        
        Args:
            name: Name of tool to remove
        """
        if name in self.tools:
            del self.tools[name]
            if name in self.tool_metadata:
                del self.tool_metadata[name]
            logger.info(f"Unregistered tool: {name}")
        else:
            logger.warning(f"Tool '{name}' not found for unregistration")
    
    def clear(self) -> None:
        """Clear all tools from registry."""
        self.tools.clear()
        self.tool_metadata.clear()
        logger.info("Cleared all tools from registry")
    
    def load_from_directory(self, directory: str, pattern: str = "*.py") -> None:
        """
        Dynamically load tools from a directory.
        
        Args:
            directory: Path to directory containing tool files
            pattern: File pattern to match (default: *.py)
        """
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.warning(f"Tool directory not found: {directory}")
            return
        
        # Find all Python files matching pattern
        tool_files = list(directory_path.glob(pattern))
        
        for tool_file in tool_files:
            try:
                self._load_tools_from_file(tool_file)
            except Exception as e:
                logger.error(f"Error loading tools from {tool_file}: {e}")
    
    def _load_tools_from_file(self, file_path: Path) -> None:
        """
        Load tools from a Python file.
        
        Args:
            file_path: Path to Python file
        """
        # Import the module
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if spec is None or spec.loader is None:
            logger.error(f"Cannot load module from {file_path}")
            return
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for tools in the module
        tools_found = 0
        
        for name, obj in inspect.getmembers(module):
            # Check for LangChain tools
            if isinstance(obj, BaseTool):
                self.register(name.lower(), obj)
                tools_found += 1
            
            # Check for functions decorated with @tool
            elif callable(obj) and hasattr(obj, '_is_langchain_tool'):
                self.register(obj.name or name, obj)
                tools_found += 1
            
            # Check for classes inheriting from BaseTool
            elif inspect.isclass(obj) and issubclass(obj, BaseTool) and obj != BaseTool:
                try:
                    tool_instance = obj()
                    tool_name = name.lower().replace('tool', '')
                    self.register(tool_name, tool_instance)
                    tools_found += 1
                except Exception as e:
                    logger.warning(f"Could not instantiate tool class {name}: {e}")
        
        if tools_found > 0:
            logger.info(f"Loaded {tools_found} tools from {file_path}")
        else:
            logger.debug(f"No tools found in {file_path}")
    
    def list_tools(self) -> Dict[str, str]:
        """
        Get dictionary mapping tool names to descriptions.
        
        Returns:
            Dictionary of {tool_name: description}
        """
        return {
            name: self.get_tool_description(name) 
            for name in self.tools.keys()
        }
    
    def __contains__(self, name: str) -> bool:
        """Check if tool exists in registry."""
        return name in self.tools
    
    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self.tools)
    
    def __iter__(self):
        """Iterate over tool names."""
        return iter(self.tools.keys())
    
    def __repr__(self) -> str:
        """String representation of registry."""
        return f"ToolRegistry({len(self.tools)} tools: {list(self.tools.keys())})"