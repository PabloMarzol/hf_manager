"""
Utility functions for working with LLMs.
"""

import json
import logging
from typing import Any, Callable, Type, TypeVar
from pydantic import BaseModel

from llm.models import get_model, ModelProvider

# Set up logging
logger = logging.getLogger(__name__)

# Type variable for generic functions
T = TypeVar('T', bound=BaseModel)


def call_llm(
    prompt,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: str,
    default_factory: Callable[[], T],
    temperature: float = 0.2,
    max_attempts: int = 3,
) -> T:
    """
    Call LLM and parse the output to a Pydantic model.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: The name of the model to use
        model_provider: The provider of the model
        pydantic_model: The Pydantic model to parse the output into
        agent_name: The name of the agent (for logging)
        default_factory: A function that returns a default model instance on failure
        temperature: The temperature to use for generation
        max_attempts: Maximum number of retry attempts
        
    Returns:
        An instance of the Pydantic model
    """
    # Convert string provider to enum
    provider = ModelProvider(model_provider)
    
    # Get the LLM
    llm = get_model(model_name, provider)
    if llm is None:
        logger.error(f"Failed to get LLM model {model_name} from {provider}")
        return default_factory()
    
    # Configure LLM parameters
    llm.temperature = temperature
    
    # Keep track of attempts
    attempts = 0
    
    while attempts < max_attempts:
        try:
            # Make the LLM call
            llm_response = llm.invoke(prompt)
            
            # Extract the content from the response
            if hasattr(llm_response, 'content'):
                response_text = llm_response.content
            else:
                response_text = str(llm_response)
            
            try:
                # Try to parse as JSON
                parsed = json.loads(response_text)
                
                # Validate with Pydantic model
                result = pydantic_model(**parsed)
                return result
                
            except json.JSONDecodeError:
                # If not JSON, try to extract JSON from the text
                logger.warning(f"{agent_name}: Failed to parse response as JSON, trying extraction")
                
                # Look for JSON-like content in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}')
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end+1]
                    try:
                        parsed = json.loads(json_str)
                        result = pydantic_model(**parsed)
                        return result
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"{agent_name}: JSON extraction failed: {e}")
                        attempts += 1
                        continue
                
                logger.error(f"{agent_name}: No valid JSON found in response")
                
            except Exception as e:
                logger.error(f"{agent_name}: Error validating response against model: {str(e)}")
            
            # If we get here, parsing failed
            attempts += 1
            
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                # Handle rate limit with exponential backoff
                wait_time = 2 ** attempts  # 2, 4, 8 seconds backoff
                logger.warning(f"{agent_name}: Rate limit reached, waiting {wait_time}s before retry ({attempts+1}/{max_attempts})")
                import time
                time.sleep(wait_time)
                attempts += 1
                continue
            else:
                logger.error(f"{agent_name}: LLM call failed: {str(e)}")
                attempts += 1
    
    # If all attempts fail, use the default
    logger.error(f"{agent_name}: Maximum attempts reached. Using default response.")
    return default_factory()