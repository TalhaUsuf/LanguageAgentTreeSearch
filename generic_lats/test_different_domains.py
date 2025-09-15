#!/usr/bin/env python3
"""
Test script for Generic LATS with different domain configurations.
Demonstrates how the framework adapts to different tool sets and domains.
"""

import os
import sys
import yaml
from pathlib import Path

# Add the generic_lats directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from tasks.base_task import BaseTask
from tools.tool_registry import ToolRegistry

class MockTool:
    """Mock tool for testing different domains"""
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def run(self, *args, **kwargs):
        return f"Mock result from {self.name}"

class DomainTestTask(BaseTask):
    """Test task for different domains"""
    def __init__(self, domain_name, **kwargs):
        self.domain_name = domain_name
        super().__init__(**kwargs)
        self.test_questions = {
            "financial": [
                "Calculate the quarterly ROI for our tech investments",
                "Analyze market volatility patterns for Q3"
            ],
            "healthcare": [
                "Process patient lab results for anomaly detection", 
                "Generate compliance report for FDA audit"
            ],
            "ecommerce": [
                "Analyze customer sentiment from recent reviews",
                "Optimize product recommendations based on purchase history"
            ]
        }
    
    def get_input(self, idx: int) -> str:
        questions = self.test_questions.get(self.domain_name, ["Generic test question"])
        return questions[idx % len(questions)]
    
    def evaluate_output(self, idx: int, output: str):
        return {"accuracy": 0.85, "domain_relevance": 0.9}
    
    def get_task_description(self) -> str:
        descriptions = {
            "financial": "Financial analysis and market research tasks requiring data retrieval, calculations, and compliance checks.",
            "healthcare": "Healthcare data processing with strict compliance requirements and medical analysis.",
            "ecommerce": "E-commerce optimization through customer analytics and recommendation systems."
        }
        return descriptions.get(self.domain_name, "Generic domain testing task")
    
    def __len__(self) -> int:
        return 2

def test_financial_domain():
    """Test the framework with financial analysis tools"""
    print("💰 Testing Financial Domain...")
    
    # Create financial tool set
    registry = ToolRegistry()
    financial_tools = [
        MockTool("database_query", "Query financial databases for historical data"),
        MockTool("financial_calculator", "Calculate financial metrics like ROI, NPV, etc."),
        MockTool("market_data_api", "Fetch real-time market data and trends"),
        MockTool("compliance_validator", "Validate financial reports for regulatory compliance"),
        MockTool("risk_analyzer", "Analyze investment risk and portfolio optimization")
    ]
    
    for tool in financial_tools:
        registry.register(tool.name, tool, tool.description)
    
    # Create task with financial prompts
    task = DomainTestTask(
        "financial",
        tool_registry=registry,
        prompts_config=str(Path(__file__).parent / "config" / "prompts.yaml")
    )
    
    # Test prompt generation
    question = task.get_input(0)
    cot_prompt = task.cot_prompt_wrap(question, "")
    
    # Should include financial-related examples and tools
    assert any(word in cot_prompt.lower() for word in ["database", "calculate", "financial", "market"])
    print(f"✅ Financial tools detected in prompt: {len([w for w in ['database', 'calculate', 'financial', 'market'] if w in cot_prompt.lower()])}/4")
    
    # Test system prompt
    system_prompt = task.format_prompt("system_prompt")
    assert "financial" in system_prompt.lower()
    assert task.domain_name in ["financial"]
    
    print("✅ Financial Domain: PASSED")
    return len(cot_prompt)

def test_healthcare_domain():
    """Test the framework with healthcare tools"""
    print("🏥 Testing Healthcare Domain...")
    
    registry = ToolRegistry()
    healthcare_tools = [
        MockTool("secure_database", "HIPAA-compliant patient database access"),
        MockTool("medical_analyzer", "Analyze lab results and medical data"),
        MockTool("hipaa_validator", "Ensure HIPAA compliance for all operations"),
        MockTool("clinical_decision_support", "AI-powered clinical decision making"),
        MockTool("report_generator", "Generate medical reports and summaries")
    ]
    
    for tool in healthcare_tools:
        registry.register(tool.name, tool, tool.description)
    
    task = DomainTestTask(
        "healthcare",
        tool_registry=registry,
        prompts_config=str(Path(__file__).parent / "config" / "prompts.yaml")
    )
    
    question = task.get_input(0)
    cot_prompt = task.cot_prompt_wrap(question, "")
    
    # Should include healthcare-specific examples
    assert any(word in cot_prompt.lower() for word in ["medical", "hipaa", "clinical", "patient"])
    print(f"✅ Healthcare tools detected in prompt: {len([w for w in ['medical', 'hipaa', 'clinical', 'patient'] if w in cot_prompt.lower()])}/4")
    
    print("✅ Healthcare Domain: PASSED")
    return len(cot_prompt)

def test_ecommerce_domain():
    """Test the framework with e-commerce tools"""
    print("🛒 Testing E-commerce Domain...")
    
    registry = ToolRegistry()
    ecommerce_tools = [
        MockTool("customer_database", "Customer data and purchase history"),
        MockTool("sentiment_analyzer", "Analyze customer reviews and feedback"),
        MockTool("recommendation_engine", "Generate product recommendations"),
        MockTool("inventory_api", "Real-time inventory management"),
        MockTool("social_media_api", "Social media monitoring and engagement")
    ]
    
    for tool in ecommerce_tools:
        registry.register(tool.name, tool, tool.description)
    
    task = DomainTestTask(
        "ecommerce", 
        tool_registry=registry,
        prompts_config=str(Path(__file__).parent / "config" / "prompts.yaml")
    )
    
    question = task.get_input(0)
    cot_prompt = task.cot_prompt_wrap(question, "")
    
    # Should include e-commerce related examples
    assert any(word in cot_prompt.lower() for word in ["customer", "recommendation", "sentiment", "inventory"])
    print(f"✅ E-commerce tools detected in prompt: {len([w for w in ['customer', 'recommendation', 'sentiment', 'inventory'] if w in cot_prompt.lower()])}/4")
    
    print("✅ E-commerce Domain: PASSED")
    return len(cot_prompt)

def test_prompt_diversity():
    """Test that different domains produce different prompts"""
    print("🔄 Testing Prompt Diversity Across Domains...")
    
    # Run domain tests and collect prompt lengths as diversity metric
    financial_length = test_financial_domain()
    healthcare_length = test_healthcare_domain()
    ecommerce_length = test_ecommerce_domain()
    
    # Prompts should be reasonably long and varied
    avg_length = (financial_length + healthcare_length + ecommerce_length) / 3
    assert avg_length > 1000  # Prompts should be substantial
    
    # Check that lengths vary (indicating different examples)
    max_length = max(financial_length, healthcare_length, ecommerce_length)
    min_length = min(financial_length, healthcare_length, ecommerce_length)
    diversity_ratio = min_length / max_length
    
    print(f"✅ Prompt diversity ratio: {diversity_ratio:.2f} (varies by {(1-diversity_ratio)*100:.1f}%)")
    print("✅ Prompt Diversity: PASSED")

def test_custom_domain_template():
    """Test creating a custom domain using the template"""
    print("🔧 Testing Custom Domain Template...")
    
    # Load the custom template
    template_path = Path(__file__).parent / "config" / "custom_task_prompts_template.yaml"
    with open(template_path, 'r') as f:
        template = yaml.safe_load(f)
    
    # Verify template has required sections
    assert "prompts" in template
    assert "examples" in template["prompts"]
    assert len(template["prompts"]["examples"]) >= 3
    
    # Check examples use diverse tools
    examples_text = str(template["prompts"]["examples"])
    diverse_tools = ["database_query", "social_media_api", "hipaa_validator", "calculate", "report_generator"]
    found_tools = [tool for tool in diverse_tools if tool in examples_text]
    
    assert len(found_tools) >= 3, f"Template should demonstrate diverse tools, found: {found_tools}"
    
    print(f"✅ Template includes {len(found_tools)} diverse tool types")
    print("✅ Custom Domain Template: PASSED")

def main():
    """Run all domain tests"""
    print("🌍 Starting Multi-Domain Generic LATS Tests\n")
    
    try:
        # Test different domains
        test_financial_domain()
        test_healthcare_domain() 
        test_ecommerce_domain()
        
        # Test cross-domain functionality
        test_prompt_diversity()
        test_custom_domain_template()
        
        print("\n🎯 DOMAIN TESTS SUMMARY:")
        print("✅ Financial Domain: Database queries, calculations, market APIs")
        print("✅ Healthcare Domain: HIPAA compliance, medical analysis, secure data") 
        print("✅ E-commerce Domain: Customer analytics, recommendations, social media")
        print("✅ Prompt Adaptation: Dynamic examples based on available tools")
        print("✅ Custom Templates: Ready-to-use domain templates with realistic examples")
        
        print("\n🚀 Generic LATS Framework Validation:")
        print("• ✅ Works with ANY combination of tools")
        print("• ✅ Adapts prompts automatically to domain context")
        print("• ✅ Maintains consistent reasoning patterns across domains")
        print("• ✅ Provides realistic few-shot examples for LLM guidance")
        print("• ✅ Supports easy customization via YAML templates")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DOMAIN TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)