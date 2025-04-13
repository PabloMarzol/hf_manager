# mixgo/utils/llm.py (updated version)
import json
import asyncio
from typing import TypeVar, Type, Optional, Any, List, Dict
from pydantic import BaseModel
import os

# Import necessary libraries
import openai
import anthropic
from langchain_groq import ChatGroq

T = TypeVar('T', bound=BaseModel)

async def call_llm(
    prompt: List[Dict[str, str]],
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory = None
) -> T:
    """
    Makes an asynchronous LLM call with retry logic.
    """
    from utils.progress import progress
    
    # Call the LLM with retries
    for attempt in range(max_retries):
        try:
            if agent_name:
                progress.update_status(agent_name, None, f"Calling LLM (attempt {attempt+1}/{max_retries})")
            
            # Call appropriate LLM based on provider
            if model_provider.lower() == "groq":
                result = await _call_groq(prompt, model_name)
            elif model_provider.lower() == "openai":
                result = await _call_openai(prompt, model_name)
            elif model_provider.lower() == "anthropic":
                result = await _call_anthropic(prompt, model_name)
            else:
                raise ValueError(f"Unsupported model provider: {model_provider}")
            
            # Parse the response into the Pydantic model
            if result:
                if agent_name:
                    progress.update_status(agent_name, None, "Processing LLM response")
                
                try:
                    model_instance = pydantic_model.model_validate(result)
                    return model_instance
                except Exception as e:
                    if attempt == max_retries - 1:
                        print(f"Failed to parse LLM response: {e}\nResponse: {result}")
                        if default_factory:
                            return default_factory()
                        return _create_default_response(pydantic_model)
            
        except Exception as e:
            if agent_name:
                progress.update_status(agent_name, None, f"Error in LLM call: {type(e).__name__}")
            
            print(f"LLM call attempt {attempt+1} failed: {e}")
            await asyncio.sleep(1)  # Wait before retry
            
            if attempt == max_retries - 1:
                print(f"All LLM call attempts failed. Using default response.")
                if default_factory:
                    return default_factory()
                return _create_default_response(pydantic_model)
    
    # Fallback - should not reach here but just in case
    if default_factory:
        return default_factory()
    return _create_default_response(pydantic_model)

async def _call_groq(messages, model_name):
    """Call Groq API."""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        client = ChatGroq(api_key=api_key)
        # Convert the messages to the expected format
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        
        response = await client.chat.completions.create(
            model=model_name or "llama-3.1-70b-versatile",
            messages=formatted_messages,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Groq API error: {e}")
        raise

async def _call_openai(messages, model_name):
    """Call OpenAI API."""
    try:
        response = await openai.chat.completions.create(
            model=model_name,
            messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise

async def _call_anthropic(messages, model_name):
    """Call Anthropic API."""
    try:
        # Convert the messages format to Anthropic's format
        system = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
        content = []
        for msg in messages:
            if msg["role"] == "user":
                content.append({"type": "text", "text": msg["content"]})
            elif msg["role"] == "assistant":
                content.append({"type": "text", "text": msg["content"]})
        
        response = await anthropic.messages.create(
            model=model_name,
            system=system,
            messages=content,
            max_tokens=1000,
        )
        return json.loads(response.content[0].text)
    except Exception as e:
        print(f"Anthropic API error: {e}")
        raise

def _create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None
    
    return model_class(**default_values)