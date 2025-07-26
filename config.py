"""
Configuration module for AutoGen Multi-Agent Research Assistant
"""
import os
from typing import List, Dict, Any

class Config:
    """Configuration class for the application"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = ""
    
    # Model Configuration
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TEMPERATURE = 0.7
    
    # Agent Configuration
    MAX_CONSECUTIVE_AUTO_REPLY = 10
    HUMAN_INPUT_MODE = "NEVER"  # Options: ALWAYS, TERMINATE, NEVER
    
    # Web Search Configuration
    MAX_SEARCH_RESULTS = 5
    
    @classmethod
    def get_llm_config(cls) -> Dict[str, Any]:
        """Get LLM configuration for AutoGen agents"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is not set.")

        config_list = [{
            "model": cls.DEFAULT_MODEL,
            "api_key": cls.OPENAI_API_KEY,
        }]
        
        return {
            "config_list": config_list,
        }
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            return False
        return True 