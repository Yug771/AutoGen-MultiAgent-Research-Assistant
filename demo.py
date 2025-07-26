"""
Demo script to test the Multi-Agent Research Assistant backend
"""
import os
from config import Config
from workflow import ResearchWorkflow
import json

def main():
    """Run a demo research session"""
    
    print("🔬 Multi-Agent Research Assistant - Demo")
    print("=" * 50)
    
    # Check configuration
    if not Config.validate_config():
        print("❌ Error: Please set your OPENAI_API_KEY in the .env file")
        print("Create a .env file with: OPENAI_API_KEY=your_api_key_here")
        return
    
    print("✅ Configuration validated")
    print(f"Using model: {Config.DEFAULT_MODEL}")
    print()
    
    # Initialize workflow
    print("Initializing research workflow...")
    try:
        llm_config = Config.get_llm_config()
        workflow = ResearchWorkflow(llm_config)
        print("✅ Workflow initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing workflow: {str(e)}")
        return
    
    # Get research topic from user
    print("\n" + "=" * 50)
    research_topic = input("Enter a research topic (or press Enter for default): ").strip()
    
    if not research_topic:
        research_topic = "Impact of artificial intelligence on healthcare"
        print(f"Using default topic: {research_topic}")
    
    # Conduct research
    print("\n🔄 Starting research... This may take a few minutes.")
    print("Agents will collaborate to research your topic.\n")
    
    try:
        result = workflow.conduct_research(
            research_topic=research_topic,
            additional_instructions="Focus on recent developments and provide specific examples."
        )
        
        print("\n✅ Research completed!")
        print("=" * 50)
        
        # Display summary
        summary = workflow.get_research_summary()
        if summary:
            print("\n📊 Research Summary:")
            print(f"- Topic: {summary['topic']}")
            print(f"- Duration: {summary['duration_seconds']:.1f} seconds")
            print(f"- Total Messages: {summary['total_messages']}")
            print(f"- Sources Found: {summary['sources_count']}")
            print(f"- Has Synthesis: {'Yes' if summary['has_synthesis'] else 'No'}")
            print(f"- Has Critique: {'Yes' if summary['has_critique'] else 'No'}")
            
            print("\n🤖 Agent Contributions:")
            for agent, count in summary['agent_contributions'].items():
                print(f"- {agent}: {count} messages")
        
        # Export report
        print("\n📄 Exporting research report...")
        report = workflow.export_research_report()
        
        # Save report to file
        filename = f"research_report_{research_topic.replace(' ', '_')[:30]}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ Report saved to: {filename}")
        
        # Also save raw data
        raw_filename = f"research_data_{research_topic.replace(' ', '_')[:30]}.json"
        with open(raw_filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"✅ Raw data saved to: {raw_filename}")
        
    except Exception as e:
        print(f"\n❌ Error during research: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Demo completed! Run 'streamlit run app.py' for the full UI experience.")

if __name__ == "__main__":
    main() 