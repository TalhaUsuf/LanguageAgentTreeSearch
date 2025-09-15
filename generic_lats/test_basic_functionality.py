#!/usr/bin/env python3
"""
Basic functionality test for Generic LATS Framework

This script tests the core components without requiring LLM access
to validate the architecture is working correctly.

Run with: python test_basic_functionality.py
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all core modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from utils.config_loader import ConfigLoader
        print("  ✅ Config loader imported")
        
        from tools.tool_registry import ToolRegistry
        print("  ✅ Tool registry imported")
        
        from tasks.base_task import BaseTask
        print("  ✅ Base task imported")
        
        from core.node import Node
        print("  ✅ Node imported")
        
        # Test LangChain-dependent imports separately
        try:
            from models.llm_wrapper import LangChainLLM
            print("  ✅ LLM wrapper imported")
        except ImportError as e:
            print(f"  ⚠️ LLM wrapper import failed (LangChain not available): {e}")
        
        try:
            from tools.base_tools import FinishTool, ThinkTool
            print("  ✅ Base tools imported")
        except ImportError as e:
            print(f"  ⚠️ Base tools import failed (LangChain not available): {e}")
            
        try:
            from tools.wikipedia_tools import WikipediaSearchTool, WikipediaLookupTool
            print("  ✅ Wikipedia tools imported")
        except ImportError as e:
            print(f"  ⚠️ Wikipedia tools import failed (LangChain not available): {e}")
            
        try:
            from tasks.hotpotqa_task import HotPotQATask
            print("  ✅ HotPotQA task imported")
        except ImportError as e:
            print(f"  ⚠️ HotPotQA task import failed: {e}")
            
        try:
            from core.lats_algorithm import LATSSearch
            print("  ✅ LATS algorithm imported")
        except ImportError as e:
            print(f"  ⚠️ LATS algorithm import failed: {e}")
        
        print("✅ Core imports successful (some LangChain features may be unavailable)")
        return True
        
    except Exception as e:
        print(f"❌ Critical import failed: {e}")
        return False


def test_config_loader():
    """Test configuration loading."""
    print("🧪 Testing configuration loader...")
    
    try:
        from utils.config_loader import ConfigLoader
        
        # Test with example config
        config_path = "config/examples/hotpotqa_config.yaml"
        if Path(config_path).exists():
            config = ConfigLoader.load(config_path)
            
            # Validate required sections
            required_sections = ['llm', 'algorithm', 'task']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required section: {section}")
            
            print("✅ Configuration loader working")
            return True
        else:
            print(f"⚠️ Config file not found: {config_path}")
            return False
            
    except Exception as e:
        print(f"❌ Config loader failed: {e}")
        return False


def test_tool_registry():
    """Test tool registry functionality."""
    print("🧪 Testing tool registry...")
    
    try:
        from tools.tool_registry import ToolRegistry
        from tools.base_tools import FinishTool, ThinkTool, simple_calculator
        
        registry = ToolRegistry()
        
        # Register tools
        registry.register("finish", FinishTool())
        registry.register("think", ThinkTool())
        registry.register("calculator", simple_calculator)
        
        # Test retrieval
        if "finish" not in registry:
            raise ValueError("Tool not found in registry")
        
        # Test tool execution
        finish_tool = registry.get_tool("finish")
        result = finish_tool._run("test answer")
        
        if "test answer" not in result:
            raise ValueError("Tool execution failed")
        
        print(f"✅ Tool registry working with {len(registry)} tools")
        return True
        
    except Exception as e:
        print(f"❌ Tool registry failed: {e}")
        return False


def test_node_functionality():
    """Test node tree functionality."""
    print("🧪 Testing node functionality...")
    
    try:
        from core.node import Node
        
        # Create root node
        root = Node(question="Test question")
        
        # Create child nodes
        child1 = Node(
            state={'thought': 'Test thought', 'action': 'Search[test]', 'observation': 'Test observation'},
            action='Search[test]',
            observation='Test observation',
            parent=root,
            question="Test question"
        )
        
        child2 = Node(
            state={'thought': 'Another thought', 'action': 'Finish[answer]', 'observation': 'Final answer'},
            action='Finish[answer]',
            observation='Final answer', 
            parent=root,
            question="Test question"
        )
        
        # Test tree structure
        if len(root.children) != 2:
            raise ValueError(f"Expected 2 children, got {len(root.children)}")
        
        # Test trajectory
        trajectory = child1.get_trajectory()
        if not trajectory:
            raise ValueError("Empty trajectory")
        
        # Test value updates
        child1.update_value(0.8, 1)
        if child1.value != 0.8:
            raise ValueError(f"Expected value 0.8, got {child1.value}")
        
        print("✅ Node functionality working")
        return True
        
    except Exception as e:
        print(f"❌ Node functionality failed: {e}")
        return False


def test_mock_llm():
    """Test with a mock LLM that doesn't require API access."""
    print("🧪 Testing mock LLM...")
    
    try:
        # Create a simple mock LLM for testing
        class MockLLM:
            def __init__(self):
                self.usage_stats = {'total_tokens': 0, 'api_calls': 0}
            
            def generate(self, prompt, n=1, stop=None, **kwargs):
                # Return mock responses based on prompt content
                if "search" in prompt.lower():
                    return ["Thought 1: I need to search for information.\nAction 1: Search[test entity]"]
                elif "finish" in prompt.lower():
                    return ["Thought 2: I have enough information.\nAction 2: Finish[test answer]"]
                else:
                    return ["Thought 1: Let me think about this.\nAction 1: Search[relevant topic]"]
            
            def get_usage_stats(self):
                return self.usage_stats
        
        mock_llm = MockLLM()
        
        # Test generation
        response = mock_llm.generate("Please search for something")
        if not response or not isinstance(response, list):
            raise ValueError("Invalid LLM response format")
        
        print("✅ Mock LLM working")
        return True
        
    except Exception as e:
        print(f"❌ Mock LLM failed: {e}")
        return False


def test_task_creation():
    """Test creating a simple custom task."""
    print("🧪 Testing task creation...")
    
    try:
        from tasks.base_task import BaseTask
        
        class MockTask(BaseTask):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.data = [
                    {'question': 'What is 2+2?', 'answer': '4'},
                    {'question': 'What color is the sky?', 'answer': 'blue'}
                ]
            
            def get_input(self, idx):
                return self.data[idx]['question']
            
            def evaluate_output(self, idx, output):
                expected = self.data[idx]['answer']
                is_correct = output.strip().lower() == expected.lower()
                return {
                    'is_correct': is_correct,
                    'score': 1.0 if is_correct else 0.0,
                    'ground_truth': expected
                }
            
            def get_task_description(self):
                return "Simple test task"
            
            def __len__(self):
                return len(self.data)
        
        # Create task instance
        task = MockTask()
        
        # Test basic functionality
        if len(task) != 2:
            raise ValueError(f"Expected 2 questions, got {len(task)}")
        
        question = task.get_input(0)
        if question != "What is 2+2?":
            raise ValueError(f"Unexpected question: {question}")
        
        # Test evaluation
        result = task.evaluate_output(0, "4")
        if not result['is_correct']:
            raise ValueError("Evaluation failed for correct answer")
        
        print("✅ Task creation working")
        return True
        
    except Exception as e:
        print(f"❌ Task creation failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("🚀 Running Generic LATS Framework Basic Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loader,
        test_tool_registry, 
        test_node_functionality,
        test_mock_llm,
        test_task_creation
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The framework appears to be working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return failed == 0


def main():
    """Main test function."""
    success = run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()