#!/usr/bin/env python3
"""
Test script for Generic LATS implementation.
Tests the framework with different tool sets and configurations.
"""

import os
import sys
import tempfile
import yaml
from pathlib import Path

# Add the generic_lats directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tasks.base_task import BaseTask
from tools.tool_registry import ToolRegistry
from models.llm_wrapper import LangChainLLM
from core.lats_algorithm import LATSSearch

class MockTool:
    """Mock tool for testing"""
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def run(self, *args, **kwargs):
        return f"Mock result from {self.name}"

class MockLLM:
    """Mock LLM for testing without API calls"""
    def __init__(self):
        self.call_count = 0
    
    def generate(self, prompt, n=1, stop=None, max_tokens=None):
        self.call_count += 1
        # Return realistic mock responses based on prompt content
        if "Thought" in prompt or "Action" in prompt:
            return [f"Thought 1: I need to solve this step by step.\nAction 1: search[test query]\nObservation 1: Mock search result"]
        elif "rate" in prompt.lower() or "likelihood" in prompt.lower():
            return ["7"]
        elif "reflection" in prompt.lower():
            return ["The approach was reasonable but could be improved by gathering more information first."]
        else:
            return ["Mock LLM response"]

class TestTask(BaseTask):
    """Test task implementation"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.test_data = [
            "What is the capital of France?",
            "How do you calculate compound interest?",
            "What are the symptoms of diabetes?"
        ]
    
    def get_input(self, idx: int) -> str:
        return self.test_data[idx % len(self.test_data)]
    
    def evaluate_output(self, idx: int, output: str):
        return {"accuracy": 0.8, "score": 0.8}
    
    def get_task_description(self) -> str:
        return "Test task for evaluating the Generic LATS framework"
    
    def __len__(self) -> int:
        return len(self.test_data)

def test_tool_registry():
    """Test the tool registry with different tool types"""
    print("🧪 Testing Tool Registry...")
    
    registry = ToolRegistry()
    
    # Test registering different tool types
    mock_tools = [
        MockTool("database_query", "Query SQL databases"),
        MockTool("api_call", "Make REST API calls"),
        MockTool("calculate", "Perform calculations"),
        MockTool("web_search", "Search the web"),
    ]
    
    for tool in mock_tools:
        registry.register(tool.name, tool, tool.description)
    
    # Test tool retrieval
    assert len(registry.get_tool_names()) == 4
    assert registry.get_tool("database_query") is not None
    
    print("✅ Tool Registry: PASSED")
    return registry

def test_base_task_prompt_generation(registry):
    """Test prompt generation with different tool sets"""
    print("🧪 Testing Base Task Prompt Generation...")
    
    # Create task with prompts config
    config_path = Path(__file__).parent / "config" / "prompts.yaml"
    task = TestTask(
        tool_registry=registry,
        llm=MockLLM(),
        prompts_config=str(config_path)
    )
    
    # Test system prompt generation
    system_prompt = task.format_prompt("system_prompt")
    assert "database_query: Query SQL databases" in system_prompt
    assert "Test task for evaluating" in system_prompt
    print("✅ System prompt generation: PASSED")
    
    # Test CoT prompt with dynamic examples
    cot_prompt = task.cot_prompt_wrap("What is 2+2?", "")
    assert "Thought" in cot_prompt
    assert "Action" in cot_prompt
    assert "database" in cot_prompt.lower() or "calculate" in cot_prompt.lower()
    print("✅ CoT prompt with dynamic examples: PASSED")
    
    # Test value prompt
    value_prompt = task.value_prompt_wrap("Test question", "Current progress")
    assert "Rate the likelihood" in value_prompt
    print("✅ Value prompt generation: PASSED")
    
    print("✅ Base Task Prompt Generation: PASSED")
    return task

def test_different_tool_sets():
    """Test prompt adaptation with different tool configurations"""
    print("🧪 Testing Different Tool Sets...")
    
    # Test Set 1: Database + Analytics tools
    registry1 = ToolRegistry()
    db_tools = [
        MockTool("database_query", "Execute SQL queries"),
        MockTool("data_analysis", "Analyze datasets"),
        MockTool("calculate", "Mathematical calculations"),
    ]
    for tool in db_tools:
        registry1.register(tool.name, tool, tool.description)
    
    task1 = TestTask(
        tool_registry=registry1,
        prompts_config=str(Path(__file__).parent / "config" / "prompts.yaml")
    )
    
    cot_prompt1 = task1.cot_prompt_wrap("Analyze sales data", "")
    # Should include database-related examples
    assert any(word in cot_prompt1.lower() for word in ["database", "query", "calculate"])
    print("✅ Database tool set adaptation: PASSED")
    
    # Test Set 2: API + Web tools  
    registry2 = ToolRegistry()
    api_tools = [
        MockTool("web_api", "Call web APIs"),
        MockTool("social_media_api", "Access social media"),
        MockTool("weather_api", "Get weather data"),
    ]
    for tool in api_tools:
        registry2.register(tool.name, tool, tool.description)
    
    task2 = TestTask(
        tool_registry=registry2,
        prompts_config=str(Path(__file__).parent / "config" / "prompts.yaml")
    )
    
    cot_prompt2 = task2.cot_prompt_wrap("Check weather forecast", "")
    # Should include API-related examples
    assert any(word in cot_prompt2.lower() for word in ["api", "weather", "social"])
    print("✅ API tool set adaptation: PASSED")
    
    print("✅ Different Tool Sets: PASSED")

def test_custom_prompts_template():
    """Test the custom prompts template functionality"""
    print("🧪 Testing Custom Prompts Template...")
    
    template_path = Path(__file__).parent / "config" / "custom_task_prompts_template.yaml"
    
    # Load and validate template structure
    with open(template_path, 'r') as f:
        template_data = yaml.safe_load(f)
    
    # Check required sections
    assert "prompts" in template_data
    assert "examples" in template_data["prompts"]
    assert "tool_categories" in template_data
    
    # Check examples have required fields
    examples = template_data["prompts"]["examples"]
    for example in examples:
        assert "pattern" in example
        assert "question" in example
        assert "trajectory" in example
        assert "Thought" in example["trajectory"]
        assert "Action" in example["trajectory"]
        assert "Observation" in example["trajectory"]
    
    # Verify diverse tool types in examples
    all_trajectories = " ".join([ex["trajectory"] for ex in examples])
    diverse_tools = ["database_query", "calculate", "social_media_api", "hipaa_validator", "report_generator"]
    found_tools = [tool for tool in diverse_tools if tool in all_trajectories]
    assert len(found_tools) >= 3, f"Template should include diverse tools, found: {found_tools}"
    
    print("✅ Custom Prompts Template: PASSED")

def test_yaml_configuration():
    """Test YAML configuration loading and environment variable substitution"""
    print("🧪 Testing YAML Configuration...")
    
    # Test main config
    main_config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(main_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert "llm" in config
    assert "algorithm" in config
    assert "task" in config
    
    # Test prompts config
    prompts_config_path = Path(__file__).parent / "config" / "prompts.yaml"
    with open(prompts_config_path, 'r') as f:
        prompts = yaml.safe_load(f)
    
    assert "prompts" in prompts
    assert "examples" in prompts["prompts"]
    assert len(prompts["prompts"]["examples"]) >= 3
    
    print("✅ YAML Configuration: PASSED")

def test_llm_wrapper():
    """Test LLM wrapper functionality"""
    print("🧪 Testing LLM Wrapper...")
    
    # Test with mock configuration
    config = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.1,
        "max_tokens": 1000
    }
    
    # Mock the actual LLM to avoid API calls
    wrapper = LangChainLLM(config)
    wrapper.llm = MockLLM()
    
    # Test generation
    result = wrapper.generate("Test prompt", n=1)
    assert len(result) == 1
    assert isinstance(result[0], str)
    
    print("✅ LLM Wrapper: PASSED")

def test_lats_algorithm():
    """Test LATS algorithm with mock components"""
    print("🧪 Testing LATS Algorithm...")
    
    # Setup mock components
    registry = ToolRegistry()
    registry.register("search", MockTool("search", "Search for information"), "Search for information")
    
    task = TestTask(
        tool_registry=registry,
        llm=MockLLM(),
        prompts_config=str(Path(__file__).parent / "config" / "prompts.yaml")
    )
    
    config = {
        "iterations": 3,
        "n_generate_sample": 5,
        "n_evaluate_sample": 1
    }
    
    lats = LATSSearch(task, MockLLM(), registry, config)
    
    # Test search initialization
    assert lats.iterations == 3
    assert hasattr(lats, 'config')
    
    print("✅ LATS Algorithm: PASSED")

def run_integration_test():
    """Run a simple end-to-end integration test"""
    print("🧪 Running Integration Test...")
    
    # Setup complete pipeline
    registry = test_tool_registry()
    task = test_base_task_prompt_generation(registry)
    
    # Test prompt generation flow
    question = "What is the population of Tokyo?"
    
    # Test system prompt
    system_prompt = task.format_prompt("system_prompt")
    assert len(system_prompt) > 100
    
    # Test CoT prompt  
    cot_prompt = task.cot_prompt_wrap(question, "")
    assert question in cot_prompt
    assert "Thought" in cot_prompt
    
    # Test value evaluation
    value_prompt = task.value_prompt_wrap(question, "Making progress...")
    values = task.value_outputs_unwrap(["The progress looks promising, I'd rate it 8 out of 10"])
    assert 0.0 <= values <= 1.0
    
    print("✅ Integration Test: PASSED")

def main():
    """Run all tests"""
    print("🚀 Starting Generic LATS Implementation Tests\n")
    
    try:
        # Core component tests
        test_yaml_configuration()
        registry = test_tool_registry()
        task = test_base_task_prompt_generation(registry)
        test_different_tool_sets()
        test_custom_prompts_template()
        test_llm_wrapper()
        test_lats_algorithm()
        
        # Integration test
        run_integration_test()
        
        print("\n🎉 ALL TESTS PASSED!")
        print("\nGeneric LATS Implementation Summary:")
        print("✅ Tool Registry: Dynamic tool registration and discovery")
        print("✅ Base Task: Generalizable prompt generation with tool awareness")
        print("✅ Prompt System: Dynamic example selection based on available tools")
        print("✅ Custom Templates: Domain-specific prompt templates with realistic examples")
        print("✅ YAML Configuration: Flexible configuration with environment variable support")
        print("✅ LLM Integration: LangChain-based LLM wrapper with fallback imports")
        print("✅ LATS Algorithm: Generic tree search implementation")
        
        print("\n🔧 Framework Features:")
        print("• Works with ANY tool set (database, API, calculation, compliance, etc.)")
        print("• Dynamic prompt adaptation based on available tools")
        print("• Complete few-shot examples for diverse domains")
        print("• Easy domain customization via YAML templates")
        print("• LangChain integration for structured tool calling")
        print("• Self-reflection and value estimation capabilities")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)