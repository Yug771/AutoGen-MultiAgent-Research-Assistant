"""
Multi-Agent Research & Synthesis Assistant - Streamlit App
"""
import streamlit as st
import os
from datetime import datetime
from config import Config
from workflow import ResearchWorkflow
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for white theme and modern UI
st.markdown("""
<style>
    /* Main theme styling */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Headers styling */
    h1 {
        color: #1f2937;
        font-weight: 700;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 10px;
        margin-bottom: 30px;
    }
    
    h2 {
        color: #374151;
        font-weight: 600;
        margin-top: 30px;
    }
    
    h3 {
        color: #4b5563;
        font-weight: 500;
    }
    
    /* Cards styling */
    .metric-card {
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-1px);
    }
    
    /* Input fields styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9fafb;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #d1fae5;
        color: #065f46;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #6ee7b7;
    }
    
    .stError {
        background-color: #fee2e2;
        color: #991b1b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #fca5a5;
    }
    
    /* Research results styling */
    .research-finding {
        background-color: #f3f4f6;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        margin: 12px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Agent message styling */
    .agent-message {
        padding: 12px;
        margin: 8px 0;
        border-radius: 8px;
        background-color: #f9fafb;
        border: 1px solid #e5e7eb;
    }
    
    .agent-name {
        font-weight: 600;
        color: #3b82f6;
        margin-bottom: 4px;
    }
    
    /* Progress indicator */
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #f9fafb;
        padding: 8px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'workflow' not in st.session_state:
    st.session_state.workflow = None
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'current_research' not in st.session_state:
    st.session_state.current_research = None
if 'api_key_validated' not in st.session_state:
    st.session_state.api_key_validated = False

# Header
st.markdown("# 🔬 AutoGen: Multi-Agent Research & Synthesis Assistant")
st.markdown("*Powered by AutoGen - Collaborative AI Agents for Comprehensive Research*")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # API Key configuration
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=Config.OPENAI_API_KEY,
        help="Enter your OpenAI API key to enable the research agents"
    )
    
    if api_key:
        Config.OPENAI_API_KEY = api_key
        if Config.validate_config():
            st.session_state.api_key_validated = True
            st.success("✅ API Key validated")
        else:
            st.error("❌ Invalid API Key")
    
    # Model selection
    model = st.selectbox(
        "Select Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo-preview", "gpt-3.5-turbo"],
        index=0,
        help="Choose the LLM model for the agents"
    )
    Config.DEFAULT_MODEL = model
    
    # Temperature
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=Config.DEFAULT_TEMPERATURE,
        step=0.1,
        help="Controls randomness in agent responses"
    )
    Config.DEFAULT_TEMPERATURE = temperature
    
    # Max rounds
    max_rounds = st.number_input(
        "Max Conversation Rounds",
        min_value=5,
        max_value=50,
        value=20,
        help="Maximum number of conversation rounds between agents"
    )
    
    st.markdown("---")
    
    # About section
    st.markdown("### 📖 About")
    st.markdown("""
    This application uses multiple AI agents to conduct comprehensive research:
    
    - **🔍 Research Agent**: Searches and gathers information
    - **📊 Synthesis Agent**: Aggregates and synthesizes findings
    - **✅ Critique Agent**: Fact-checks and evaluates quality
    - **👤 User Proxy**: Manages human interaction
    """)

# Main content area
if not st.session_state.api_key_validated:
    st.warning("⚠️ Please enter a valid OpenAI API key in the sidebar to begin.")
else:
    # Initialize workflow if needed
    if st.session_state.workflow is None:
        try:
            llm_config = Config.get_llm_config()
            st.session_state.workflow = ResearchWorkflow(llm_config)
            st.success("✅ Research workflow initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing workflow: {str(e)}")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔬 New Research", "📚 History", "📊 Current Research", "📈 Analytics"])
    
    with tab1:
        st.markdown("## Start New Research")
        
        # Research form
        with st.form("research_form"):
            research_topic = st.text_input(
                "Research Topic",
                placeholder="e.g., Impact of AI on healthcare",
                help="Enter the topic you want the agents to research"
            )
            
            additional_instructions = st.text_area(
                "Additional Instructions (Optional)",
                placeholder="e.g., Focus on recent developments, include statistics, compare different perspectives",
                height=100,
                help="Provide any specific instructions for the research"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                submit_button = st.form_submit_button("🚀 Start Research", use_container_width=True)
        
        if submit_button and research_topic:
            with st.spinner("🔄 Conducting research... This may take a few minutes."):
                try:
                    # Reset the workflow to ensure a clean state for the new research.
                    st.session_state.workflow.reset()

                    # Create placeholders for live updates
                    message_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    status_placeholder = st.empty()
                    
                    try:
                        # Show initial status
                        status_placeholder.info("🔄 Initializing research agents...")
                        progress_bar.progress(25)
                        
                        # Conduct research
                        research_result = st.session_state.workflow.conduct_research(
                            research_topic=research_topic,
                            additional_instructions=additional_instructions
                        )
                        
                        # Update progress based on result
                        if research_result and not research_result.get('error'):
                            progress_bar.progress(100)
                            status_placeholder.success("✅ Research completed successfully!")
                            
                            # Store in session state
                            st.session_state.current_research = research_result
                            st.session_state.research_history.append(research_result)
                            
                            # Show success animation
                            st.balloons()
                            
                            # Switch to results tab
                            st.info("📊 Switch to the 'Current Research' tab to view the results.")
                        else:
                            progress_bar.progress(100)
                            status_placeholder.error("❌ Research failed or was interrupted.")
                            
                    except Exception as e:
                        # Clear the progress bar and status on error
                        progress_bar.progress(100)
                        status_placeholder.error(f"❌ Error during research: {str(e)}")
                        logger.error(f"Research error: {str(e)}")
                        
                    finally:
                        # Clear the progress bar to indicate completion
                        progress_bar.empty()
                    
                except Exception as e:
                    st.error(f"❌ Error during research: {str(e)}")
                    logger.error(f"Research error: {str(e)}")
    
    with tab2:
        st.markdown("## 📚 Research History")
        
        if st.session_state.research_history:
            # Display research history in reverse chronological order
            for idx, research in enumerate(reversed(st.session_state.research_history)):
                with st.expander(f"🔬 {research.get('topic', 'Unknown Topic')} - {research.get('timestamp', 'Unknown Time')[:19]}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Duration", f"{research.get('duration_seconds', 0):.1f}s")
                    with col2:
                        st.metric("Messages", len(research.get('messages', [])))
                    with col3:
                        st.metric("Sources", len(research.get('results', {}).get('sources', [])))
                    
                    # View and export buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"View Research #{len(st.session_state.research_history) - idx}", key=f"view_{idx}"):
                            st.session_state.current_research = research
                            st.info("📊 Switch to the 'Current Research' tab to view the results.")
                    
                    with col2:
                        # Export research report
                        report = st.session_state.workflow.export_research_report(-(idx + 1))
                        st.download_button(
                            label="📥 Download Report",
                            data=report,
                            file_name=f"research_report_{research.get('timestamp', 'unknown')[:19].replace(':', '-')}.md",
                            mime="text/markdown",
                            key=f"download_{idx}"
                        )
        else:
            st.info("No research history yet. Start a new research to begin!")
    
    with tab3:
        st.markdown("## 📊 Current Research Results")
        
        if st.session_state.current_research and 'error' not in st.session_state.current_research:
            research = st.session_state.current_research
            results = research.get('results', {})
            
            # Research summary
            st.markdown(f"### Topic: {research.get('topic', 'Unknown')}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{research.get('duration_seconds', 0):.1f}s")
            with col2:
                st.metric("Total Messages", len(research.get('messages', [])))
            with col3:
                st.metric("Sources Found", len(results.get('sources', [])))
            with col4:
                st.metric("Findings", len(results.get('findings', [])))
            
            # Executive Summary
            if results.get('synthesis'):
                st.markdown("### 📋 Executive Summary")
                st.markdown(f'<div class="research-finding">{results["synthesis"]}</div>', unsafe_allow_html=True)
            
            # Detailed Findings
            if results.get('findings'):
                st.markdown("### 🔍 Detailed Findings")
                for i, finding in enumerate(results['findings'], 1):
                    with st.expander(f"Finding {i} - by {finding.get('agent', 'Unknown')}"):
                        st.markdown(finding.get('content', ''))
            
            # Critical Analysis
            if results.get('critique'):
                st.markdown("### ✅ Critical Analysis")
                st.markdown(f'<div class="research-finding">{results["critique"]}</div>', unsafe_allow_html=True)
            
            # Sources
            if results.get('sources'):
                st.markdown("### 📚 Sources")
                for source in results['sources']:
                    st.markdown(f"- [{source}]({source})")
            
            # Full Conversation
            with st.expander("💬 View Full Agent Conversation"):
                for msg in research.get('messages', []):
                    st.markdown(f'<div class="agent-message"><div class="agent-name">{msg.get("name", "Unknown")}</div>{msg.get("content", "")}</div>', unsafe_allow_html=True)
            
            # Export options
            st.markdown("### 📥 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                report = st.session_state.workflow.export_research_report()
                st.download_button(
                    label="📄 Download Research Report (Markdown)",
                    data=report,
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            
            with col2:
                st.download_button(
                    label="📊 Download Raw Data (JSON)",
                    data=json.dumps(research, indent=2),
                    file_name=f"research_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        elif st.session_state.current_research and 'error' in st.session_state.current_research:
            st.error(f"❌ Research failed: {st.session_state.current_research['error']}")
        else:
            st.info("No current research to display. Start a new research or select one from history.")
    
    with tab4:
        st.markdown("## 📈 Research Analytics")
        
        if st.session_state.research_history:
            # Overall statistics
            st.markdown("### 📊 Overall Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            total_researches = len(st.session_state.research_history)
            total_duration = sum(r.get('duration_seconds', 0) for r in st.session_state.research_history)
            avg_duration = total_duration / total_researches if total_researches > 0 else 0
            total_messages = sum(len(r.get('messages', [])) for r in st.session_state.research_history)
            
            with col1:
                st.metric("Total Researches", total_researches)
            with col2:
                st.metric("Total Duration", f"{total_duration:.1f}s")
            with col3:
                st.metric("Average Duration", f"{avg_duration:.1f}s")
            with col4:
                st.metric("Total Messages", total_messages)
            
            # Agent participation
            st.markdown("### 🤖 Agent Participation")
            
            agent_stats = {}
            for research in st.session_state.research_history:
                for agent, count in research.get('agent_contributions', {}).items():
                    agent_stats[agent] = agent_stats.get(agent, 0) + count
            
            if agent_stats:
                # Create columns for agent stats
                cols = st.columns(len(agent_stats))
                for idx, (agent, count) in enumerate(agent_stats.items()):
                    with cols[idx]:
                        st.metric(agent.replace('_', ' ').title(), count)
            
            # Recent topics
            st.markdown("### 📝 Recent Research Topics")
            
            for research in st.session_state.research_history[-5:]:
                st.markdown(f"- **{research.get('topic', 'Unknown')}** - {research.get('timestamp', '')[:19]}")
        
        else:
            st.info("No analytics data available yet. Complete some research to see analytics!")

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #6b7280; padding: 20px;">'
    'Built ❤️ by Yug Patel | '
    '<a href="https://github.com/yug771" style="color: #3b82f6;">GitHub</a> | '
    '<a href="https://www.linkedin.com/in/027-yug-patel/" style="color: #3b82f6;">LinkedIn</a>'
    '</div>',
    unsafe_allow_html=True
) 