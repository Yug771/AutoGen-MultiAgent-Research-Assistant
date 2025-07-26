"""
Custom AutoGen agents for the Multi-Agent Research Assistant
"""
import autogen
from typing import Dict, Any, List, Optional, Union
from web_search import WebSearcher
import json
import logging

logger = logging.getLogger(__name__)


class ResearchAgent(autogen.AssistantAgent):
    """
    Research Agent that performs web searches and gathers information
    """
    
    def __init__(self, name: str = "research_agent", llm_config: Optional[Dict] = None, **kwargs):
        # Initialize web searcher
        self.web_searcher = WebSearcher(max_results=kwargs.pop("max_search_results", 5))
        
        # Create system message for the research agent
        system_message = """You are a Research Agent specialized in gathering information from the web.
        
Your responsibilities:
1. Search for relevant information based on the research topic
2. Extract key findings from web sources
3. Provide accurate citations and sources
4. Focus on credible and recent information
5. Present findings in a structured format

When you need to search the web, use the search_web function.
When you need to extract content from a specific URL, use the extract_content function.
Always provide the source URL for any information you present."""
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs
        )
        
        # Register functions for web search
        # For pyautogen 0.1.14, we use register_function
        self.register_function(
            function_map={
                "search_web": self._search_web,
                "extract_content": self._extract_content,
            }
        )
    
    def _search_web(self, query: str) -> str:
        """
        Search the web for information
        
        Args:
            query: Search query
            
        Returns:
            JSON string of search results
        """
        results = self.web_searcher.search(query)
        return json.dumps(results, indent=2)
    
    def _extract_content(self, url: str) -> str:
        """
        Extract content from a specific URL
        
        Args:
            url: URL to extract content from
            
        Returns:
            JSON string of extracted content
        """
        content = self.web_searcher.extract_article_content(url)
        return json.dumps(content, indent=2)


class SynthesisAgent(autogen.AssistantAgent):
    """
    Synthesis Agent that aggregates and synthesizes research findings
    """
    
    def __init__(self, name: str = "synthesis_agent", llm_config: Optional[Dict] = None, **kwargs):
        system_message = """You are a Synthesis Agent specialized in aggregating and synthesizing research findings.
        
Your responsibilities:
1. Combine information from multiple sources into coherent insights
2. Identify patterns and connections between different findings
3. Create structured summaries and reports
4. Highlight key takeaways and conclusions
5. Organize information in a logical and accessible manner

Focus on creating comprehensive yet concise syntheses that capture the essence of the research.
Always maintain objectivity and acknowledge different perspectives when present."""
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs
        )


class CritiqueAgent(autogen.AssistantAgent):
    """
    Critique Agent that fact-checks and evaluates research quality
    """
    
    def __init__(self, name: str = "critique_agent", llm_config: Optional[Dict] = None, **kwargs):
        system_message = """You are a Critique Agent specialized in fact-checking and quality evaluation.
        
Your responsibilities:
1. Verify the accuracy of presented information
2. Check the credibility of sources
3. Identify potential biases or limitations
4. Suggest areas that need further research
5. Ensure logical consistency in arguments
6. Point out any gaps or weaknesses in the research

Be constructive in your critique, suggesting improvements rather than just pointing out flaws.
Always explain your reasoning when questioning information or sources."""
        
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs
        )


class CustomUserProxyAgent(autogen.UserProxyAgent):
    """
    Custom UserProxyAgent with enhanced capabilities for the research assistant
    """
    
    def __init__(self, name: str = "user_proxy", **kwargs):
        # Set default code execution config
        if "code_execution_config" not in kwargs:
            kwargs["code_execution_config"] = False
        
        super().__init__(name=name, **kwargs)


def create_research_team(llm_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create the research team with all agents
    
    Args:
        llm_config: LLM configuration for the agents
        
    Returns:
        Dictionary containing all agents
    """
    # Clean llm_config to remove any unexpected parameters
    clean_llm_config = {
        "config_list": llm_config.get("config_list", []),
    }
    
    # Create agents
    user_proxy = CustomUserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    )
    
    research_agent = ResearchAgent(
        name="research_agent",
        llm_config=clean_llm_config,
        max_consecutive_auto_reply=10,
    )
    
    synthesis_agent = SynthesisAgent(
        name="synthesis_agent",
        llm_config=clean_llm_config,
        max_consecutive_auto_reply=10,
    )
    
    critique_agent = CritiqueAgent(
        name="critique_agent",
        llm_config=clean_llm_config,
        max_consecutive_auto_reply=10,
    )
    
    return {
        "user_proxy": user_proxy,
        "research_agent": research_agent,
        "synthesis_agent": synthesis_agent,
        "critique_agent": critique_agent,
    } 