"""
Workflow manager for orchestrating multi-agent collaboration
"""
import autogen
from typing import Dict, Any, List, Optional
from agents import create_research_team
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ResearchWorkflow:
    """
    Manages the research workflow with multiple agents
    """
    
    def __init__(self, llm_config: Dict[str, Any]):
        """
        Initialize the research workflow
        
        Args:
            llm_config: LLM configuration for agents
        """
        self.llm_config = llm_config
        self.agents = create_research_team(llm_config)
        self.research_history = []
        
        # Create group chat manager (pyautogen 0.1.14 API)
        self.group_chat = autogen.GroupChat(
            agents=[
                self.agents["user_proxy"],
                self.agents["research_agent"],
                self.agents["synthesis_agent"],
                self.agents["critique_agent"]
            ],
            messages=[],
            max_round=20
        )
        
        # Create a manager config without proxy settings
        manager_llm_config = llm_config.copy()
        if 'proxies' in manager_llm_config:
            del manager_llm_config['proxies']
            
        self.manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config=manager_llm_config,
        )

    def reset(self):
        """Resets the conversation history for a new research task."""
        self.group_chat.messages.clear()
        # It's also good practice to reset the manager to ensure a clean state.
        self.manager = autogen.GroupChatManager(
            groupchat=self.group_chat,
            llm_config=self.llm_config,
        )
        logger.info("Research workflow has been reset.")

    def conduct_research(self, research_topic: str, additional_instructions: str = "") -> Dict[str, Any]:
        """
        Conduct research on a given topic
        
        Args:
            research_topic: The topic to research
            additional_instructions: Any additional instructions for the research
            
        Returns:
            Dictionary containing research results and metadata
        """
        start_time = datetime.now()
        
        # Prepare the initial message
        initial_message = f"""Please conduct comprehensive research on the following topic:

**Topic**: {research_topic}

{additional_instructions}

**Research Process**:
1. The Research Agent should search for relevant information from credible sources
2. The Synthesis Agent should aggregate and synthesize the findings
3. The Critique Agent should fact-check and evaluate the quality of the research
4. Continue the discussion until a comprehensive research deliverable is ready

Please begin the research process."""
        
        try:
            # Initiate the group chat
            self.agents["user_proxy"].initiate_chat(
                self.manager,
                message=initial_message,
            )
            
            # Extract results from the conversation
            messages = self.group_chat.messages
            research_results = self._extract_research_results(messages)
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Create research record
            research_record = {
                "topic": research_topic,
                "timestamp": start_time.isoformat(),
                "duration_seconds": duration,
                "messages": messages,
                "results": research_results,
                "agent_contributions": self._analyze_agent_contributions(messages),
            }
            
            # Save to history
            self.research_history.append(research_record)
            
            return research_record
            
        except Exception as e:
            logger.error(f"Error during research: {str(e)}")
            return {
                "error": str(e),
                "topic": research_topic,
                "timestamp": start_time.isoformat(),
            }
    
    def _extract_research_results(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract structured research results from conversation messages
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Dictionary containing extracted research results
        """
        results = {
            "findings": [],
            "sources": [],
            "synthesis": "",
            "critique": "",
            "recommendations": [],
        }
        
        for message in messages:
            content = message.get("content", "")
            name = message.get("name", "")
            
            # Extract findings from research agent
            if name == "research_agent":
                if "http" in content:  # Extract URLs as sources
                    import re
                    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
                    results["sources"].extend(urls)
                
                # Add content as findings
                if len(content) > 50:  # Skip very short messages
                    results["findings"].append({
                        "content": content,
                        "agent": name,
                    })
            
            # Extract synthesis
            elif name == "synthesis_agent" and len(content) > 100:
                results["synthesis"] = content
            
            # Extract critique
            elif name == "critique_agent" and len(content) > 100:
                results["critique"] = content
        
        # Remove duplicate sources
        results["sources"] = list(set(results["sources"]))
        
        return results
    
    def _analyze_agent_contributions(self, messages: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Analyze how many times each agent contributed
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Dictionary with agent contribution counts
        """
        contributions = {}
        
        for message in messages:
            agent_name = message.get("name", "unknown")
            contributions[agent_name] = contributions.get(agent_name, 0) + 1
        
        return contributions
    
    def get_research_summary(self, index: int = -1) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a specific research session
        
        Args:
            index: Index of the research session (default: -1 for most recent)
            
        Returns:
            Research summary or None if not found
        """
        if not self.research_history or abs(index) > len(self.research_history):
            return None
        
        research = self.research_history[index]
        
        return {
            "topic": research["topic"],
            "timestamp": research["timestamp"],
            "duration_seconds": research["duration_seconds"],
            "total_messages": len(research["messages"]),
            "agent_contributions": research["agent_contributions"],
            "sources_count": len(research["results"]["sources"]),
            "has_synthesis": bool(research["results"]["synthesis"]),
            "has_critique": bool(research["results"]["critique"]),
        }
    
    def export_research_report(self, index: int = -1) -> str:
        """
        Export a research session as a formatted report
        
        Args:
            index: Index of the research session (default: -1 for most recent)
            
        Returns:
            Formatted research report as string
        """
        if not self.research_history or abs(index) > len(self.research_history):
            return "No research found."
        
        research = self.research_history[index]
        results = research["results"]
        
        report = f"""# Research Report

**Topic**: {research["topic"]}
**Date**: {research["timestamp"]}
**Duration**: {research["duration_seconds"]:.1f} seconds

## Executive Summary

{results["synthesis"] if results["synthesis"] else "No synthesis available."}

## Detailed Findings

"""
        
        for i, finding in enumerate(results["findings"], 1):
            report += f"### Finding {i}\n\n{finding['content']}\n\n"
        
        if results["sources"]:
            report += "## Sources\n\n"
            for source in results["sources"]:
                report += f"- {source}\n"
        
        if results["critique"]:
            report += f"\n## Critical Analysis\n\n{results['critique']}\n"
        
        report += f"\n## Methodology\n\nThis research was conducted using a multi-agent system with {len(research['agent_contributions'])} participating agents.\n"
        
        return report 