"""
Configuration loader with YAML support and environment variable substitution.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and processes YAML configuration files with environment variable substitution."""
    
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file with environment variable substitution.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the loaded configuration
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML is malformed
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Substitute environment variables
        content = ConfigLoader._substitute_env_vars(content)
        
        try:
            config = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML config: {e}")
        
        # Validate required sections
        ConfigLoader._validate_config(config)
        
        logger.info("Configuration loaded successfully")
        return config
    
    @staticmethod
    def _substitute_env_vars(content: str) -> str:
        """
        Substitute environment variables in the format ${VAR_NAME} or ${VAR_NAME:default}.
        
        Args:
            content: String content with environment variable placeholders
            
        Returns:
            String with environment variables substituted
        """
        def replace_var(match):
            var_expr = match.group(1)
            
            # Handle default values: ${VAR_NAME:default_value}
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                var_name = var_expr.strip()
                value = os.getenv(var_name)
                if value is None:
                    logger.warning(f"Environment variable {var_name} not found, leaving placeholder")
                    return match.group(0)  # Return original placeholder
                return value
        
        # Pattern matches ${VAR_NAME} or ${VAR_NAME:default}
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_var, content)
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """
        Validate that required configuration sections exist.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValueError: If required sections are missing
        """
        required_sections = ['llm', 'algorithm', 'task']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section '{section}' is missing")
        
        # Validate LLM configuration
        llm_config = config['llm']
        required_llm_fields = ['provider', 'model']
        
        for field in required_llm_fields:
            if field not in llm_config:
                raise ValueError(f"Required LLM configuration field '{field}' is missing")
        
        # Validate algorithm configuration
        algo_config = config['algorithm']
        required_algo_fields = ['type', 'iterations']
        
        for field in required_algo_fields:
            if field not in algo_config:
                raise ValueError(f"Required algorithm configuration field '{field}' is missing")
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries, with override_config taking precedence.
        
        Args:
            base_config: Base configuration dictionary
            override_config: Configuration dictionary with override values
            
        Returns:
            Merged configuration dictionary
        """
        def deep_merge(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            
            return result
        
        return deep_merge(base_config, override_config)
    
    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str) -> None:
        """
        Save a configuration dictionary to a YAML file.
        
        Args:
            config: Configuration dictionary to save
            output_path: Path where to save the configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to: {output_path}")


class ConfigValidator:
    """Validates configuration values and types."""
    
    @staticmethod
    def validate_llm_config(llm_config: Dict[str, Any]) -> None:
        """Validate LLM configuration parameters."""
        
        # Validate provider
        valid_providers = ['openai', 'openai-compatible', 'anthropic', 'langchain-custom']
        provider = llm_config.get('provider')
        if provider not in valid_providers:
            raise ValueError(f"Invalid LLM provider '{provider}'. Valid options: {valid_providers}")
        
        # Validate temperature
        temperature = llm_config.get('temperature', 1.0)
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temperature}")
        
        # Validate max_tokens
        max_tokens = llm_config.get('max_tokens', 500)
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive integer, got {max_tokens}")
    
    @staticmethod
    def validate_algorithm_config(algo_config: Dict[str, Any]) -> None:
        """Validate algorithm configuration parameters."""
        
        # Validate algorithm type
        valid_algorithms = ['lats', 'tot', 'rap']
        algo_type = algo_config.get('type')
        if algo_type not in valid_algorithms:
            raise ValueError(f"Invalid algorithm type '{algo_type}'. Valid options: {valid_algorithms}")
        
        # Validate iterations
        iterations = algo_config.get('iterations', 30)
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError(f"iterations must be a positive integer, got {iterations}")
        
        # Validate generation samples
        n_generate = algo_config.get('n_generate_sample', 5)
        if not isinstance(n_generate, int) or n_generate <= 0:
            raise ValueError(f"n_generate_sample must be a positive integer, got {n_generate}")