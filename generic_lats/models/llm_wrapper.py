"""
LangChain-based LLM wrapper for Generic LATS framework.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import time
import backoff
import openai
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
        from langchain.schema import HumanMessage, SystemMessage
    except ImportError:
        # Fallback for basic testing
        ChatOpenAI = None
        HumanMessage = None
        SystemMessage = None
from langchain.callbacks.base import BaseCallbackHandler

logger = logging.getLogger(__name__)


class TokenTrackingCallback(BaseCallbackHandler):
    """Callback to track token usage during LLM calls."""
    
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        self.call_count += 1
        
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends running."""
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens += token_usage.get('prompt_tokens', 0)
            self.completion_tokens += token_usage.get('completion_tokens', 0)
            self.total_tokens += token_usage.get('total_tokens', 0)
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get current token usage statistics."""
        return {
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'api_calls': self.call_count
        }
    
    def reset(self) -> None:
        """Reset all counters."""
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0


class LangChainLLM:
    """
    LangChain-based LLM wrapper with support for multiple providers and structured outputs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LLM wrapper.
        
        Args:
            config: LLM configuration dictionary with provider, model, etc.
        """
        self.config = config
        self.provider = config.get('provider', 'openai-compatible')
        self.model = config.get('model', 'gpt-3.5-turbo')
        self.temperature = config.get('temperature', 1.0)
        self.max_tokens = config.get('max_tokens', 500)
        self.timeout = config.get('timeout', 30)
        
        # Token tracking
        self.callback = TokenTrackingCallback()
        
        # Initialize the LangChain model
        self.llm = self._initialize_llm()
        
        logger.info(f"Initialized LLM: provider={self.provider}, model={self.model}")
    
    def _initialize_llm(self) -> ChatOpenAI:
        """Initialize the appropriate LangChain LLM based on provider."""
        
        if self.provider in ['openai', 'openai-compatible']:
            # Use ChatOpenAI for both OpenAI and compatible APIs
            llm_kwargs = {
                'model': self.model,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'callbacks': [self.callback],
                'request_timeout': self.timeout,
            }
            
            # Add API key if provided
            api_key = self.config.get('api_key')
            if api_key:
                llm_kwargs['api_key'] = api_key
            
            # Add base URL for compatible APIs
            base_url = self.config.get('base_url')
            if base_url and self.provider == 'openai-compatible':
                llm_kwargs['base_url'] = base_url
            
            return ChatOpenAI(**llm_kwargs)
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    @backoff.on_exception(
        backoff.expo,
        (openai.APIConnectionError, openai.RateLimitError, openai.APITimeoutError),
        max_tries=3,
        base=2.0
    )
    def generate(
        self, 
        prompt: str, 
        system_message: Optional[str] = None,
        n: int = 1,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate text responses from the LLM.
        
        Args:
            prompt: The input prompt
            system_message: Optional system message
            n: Number of responses to generate
            stop: Stop sequences
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated response strings
        """
        start_time = time.time()
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        # Update parameters
        generation_params = {
            'temperature': kwargs.get('temperature', self.temperature),
            'max_tokens': kwargs.get('max_tokens', self.max_tokens),
        }
        
        if stop:
            generation_params['stop'] = stop
        
        responses = []
        
        # Generate multiple responses if requested
        for _ in range(n):
            try:
                # Create a new LLM instance with updated parameters
                # import pdb; pdb.set_trace()
                temp_llm = self.llm.__class__(
                    **{**self.llm.__dict__, **generation_params, 'callbacks': [self.callback]}
                )
                
                response = temp_llm(messages)
                responses.append(response.content.strip())
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                responses.append("")  # Empty response on error
        
        generation_time = time.time() - start_time
        logger.debug(f"Generated {len(responses)} responses in {generation_time:.2f}s")
        
        return responses
    
    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Any],
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate a response with tool calling capabilities.
        
        Args:
            prompt: The input prompt
            tools: List of LangChain tools
            system_message: Optional system message
            
        Returns:
            Generated response string
        """
        # Bind tools to the LLM
        llm_with_tools = self.llm.bind_tools(tools)
        
        # Prepare messages
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))
        
        try:
            response = llm_with_tools.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating response with tools: {e}")
            return ""
    
    def get_usage_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get token usage and cost statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        stats = self.callback.get_usage_stats()
        
        # Add cost estimation (rough approximation)
        cost = 0.0
        if self.model.startswith('gpt-4'):
            cost = (stats['prompt_tokens'] * 0.03 + stats['completion_tokens'] * 0.06) / 1000
        elif self.model.startswith('gpt-3.5'):
            cost = (stats['prompt_tokens'] * 0.001 + stats['completion_tokens'] * 0.002) / 1000
        
        stats['estimated_cost'] = cost
        return stats
    
    def reset_usage_stats(self) -> None:
        """Reset token usage statistics."""
        self.callback.reset()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update LLM configuration parameters.
        
        Args:
            new_config: Dictionary with new configuration values
        """
        self.config.update(new_config)
        self.temperature = self.config.get('temperature', self.temperature)
        self.max_tokens = self.config.get('max_tokens', self.max_tokens)
        
        # Reinitialize LLM if critical parameters changed
        critical_params = ['provider', 'model', 'base_url', 'api_key']
        if any(param in new_config for param in critical_params):
            logger.info("Critical parameters changed, reinitializing LLM...")
            self.llm = self._initialize_llm()


# Convenience function for backward compatibility
def gpt(prompt: str, model: str = "gpt-3.5-turbo", temperature: float = 1.0, 
        max_tokens: int = 100, n: int = 1, stop: Optional[List[str]] = None) -> List[str]:
    """
    Backward compatibility function that mimics the original gpt() interface.
    
    Args:
        prompt: Input prompt
        model: Model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        n: Number of responses
        stop: Stop sequences
        
    Returns:
        List of generated responses
    """
    config = {
        'provider': 'openai-compatible',
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    
    # Add environment variables if available
    import os
    if 'OPENAI_API_BASE' in os.environ:
        config['base_url'] = os.environ['OPENAI_API_BASE']
    if 'OPENAI_API_KEY' in os.environ:
        config['api_key'] = os.environ['OPENAI_API_KEY']
    
    llm = LangChainLLM(config)
    return llm.generate(prompt, n=n, stop=stop)


# Global instance for backward compatibility
_global_llm_instance: Optional[LangChainLLM] = None

def get_global_llm() -> LangChainLLM:
    """Get or create global LLM instance."""
    global _global_llm_instance
    
    if _global_llm_instance is None:
        import os
        config = {
            'provider': 'openai-compatible',
            'model': os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            'base_url': os.getenv('OPENAI_API_BASE', ''),
            'api_key': os.getenv('OPENAI_API_KEY', ''),
            'temperature': 1.0,
            'max_tokens': 500,
        }
        _global_llm_instance = LangChainLLM(config)
    
    return _global_llm_instance