# llm/client.py
from typing import Any, Dict, Type, TypeVar
from pydantic import BaseModel
from llm.models import get_model, ModelProvider

T = TypeVar('T', bound=BaseModel)

class LLMClient:
    """Client for interacting with LLM services."""
    
    def __init__(self, model_name: str, model_provider: str):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name of the LLM model to use
            model_provider: Provider of the LLM model (e.g., OpenAI)
        """
        self.model_name = model_name
        self.model_provider = model_provider
        self.llm = get_model(model_name, model_provider)
    
    async def generate_decision(
        self, 
        context: Dict[str, Any], 
        system_prompt: str,
        output_model: Type[T]
    ) -> T:
        """
        Generate a structured decision using the LLM.
        
        Args:
            context: Context information for the decision
            system_prompt: System prompt for the LLM
            output_model: Pydantic model for the output
            
        Returns:
            Instance of the output model
        """
        # Format the prompt with context info
        user_prompt = (
            f"Please analyze the following trading signals and make a decision for {context['ticker']}.\n\n"
            f"Signals: {context['signals']}\n\n"
            f"Current Position: {context['position']}\n\n"
            f"Available Cash: ${context['cash']:,.2f}\n\n"
            f"Portfolio Context: {context['portfolio_context']}\n\n"
            f"Generate a trading decision with action, quantity, confidence, and reasoning."
        )
        
        # Use with_structured_output for supported models
        structured_llm = self.llm.with_structured_output(output_model)
        
        # Invoke the LLM with retry logic
        from utils.llm import call_llm
        return call_llm(
            prompt=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model_name=self.model_name,
            model_provider=self.model_provider,
            pydantic_model=output_model,
            agent_name="mixgo_agent",
            max_retries=3
        )