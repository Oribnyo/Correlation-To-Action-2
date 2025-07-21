from datetime import datetime
import os

def create_markdown_report(llm_response, prompt_filename):
    """Create Markdown document with LLM response"""
    # Remove file extension for cleaner naming
    prompt_name = prompt_filename.replace('.txt', '')
    
    # Create markdown content
    markdown_content = f"""# AI-Assisted Operational Insight Report - {prompt_name}

**Generated on:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## LLM Response

{llm_response}

---
*Report generated automatically by AI system*
"""
    
    # Save document
    filename = f'AI-Assisted Operational Insight Report - {prompt_name}.md'
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"✅ Report saved as: {filename}")
    return filename

def save_llm_response_to_word(llm_response, prompt_filename="prompt.txt"):
    """Main function to save LLM response to Markdown document"""
    return create_markdown_report(llm_response, prompt_filename) 