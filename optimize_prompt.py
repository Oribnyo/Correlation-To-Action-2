#!/usr/bin/env python3
"""
Prompt Optimization System for prompt_with_process_cheatsheet.txt
Based on OpenAI Cookbook: https://cookbook.openai.com/examples/optimize_prompts
"""

import asyncio
import json
import os
from enum import Enum
from typing import Any, List, Dict
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Role(str, Enum):
    """Role enum for chat messages."""
    user = "user"
    assistant = "assistant"

class ChatMessage(BaseModel):
    """Single chat message used in few-shot examples."""
    role: Role
    content: str

class Issues(BaseModel):
    """Structured output returned by checkers."""
    has_issues: bool
    issues: List[str]
    
    @classmethod
    def no_issues(cls) -> "Issues":
        return cls(has_issues=False, issues=[])

class FewShotIssues(Issues):
    """Output for few-shot contradiction detector including optional rewrite suggestions."""
    rewrite_suggestions: List[str] = Field(default_factory=list)
    
    @classmethod
    def no_issues(cls) -> "FewShotIssues":
        return cls(has_issues=False, issues=[], rewrite_suggestions=[])

class MessagesOutput(BaseModel):
    """Structured output returned by `rewrite_messages_agent`."""
    messages: list[ChatMessage]

class DevRewriteOutput(BaseModel):
    """Rewriter returns the cleaned-up developer prompt."""
    new_developer_message: str

async def check_contradictions(prompt: str) -> Issues:
    """Check for contradictions in the prompt."""
    response = await client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """You are Dev-Contradiction-Checker.

Goal: Detect genuine self-contradictions or impossibilities inside the developer prompt.

Definitions:
• A contradiction = two clauses that cannot both be followed.
• Overlaps or redundancies are NOT contradictions.

What you MUST do:
1. Compare every imperative / prohibition against all others.
2. List at most FIVE contradictions (each as ONE bullet).
3. If no contradiction exists, say so.

Output format (strict JSON):
Return only an object that matches this schema:
{"has_issues": <bool>, "issues": ["<bullet 1>", "<bullet 2>"]}
- has_issues = true IFF the issues array is non-empty.
- Do not add extra keys, comments or markdown."""
            },
            {
                "role": "user",
                "content": f"Analyze this prompt for contradictions:\n\n{prompt}"
            }
        ]
    )
    
    result = json.loads(response.choices[0].message.content)
    return Issues(**result)

async def check_format_issues(prompt: str) -> Issues:
    """Check for format specification issues."""
    response = await client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """You are Format-Checker.

Task: Decide whether the developer prompt requires a structured output (JSON/CSV/XML/Markdown table, etc.).
If so, flag any missing or unclear aspects of that format.

Steps:
1. Categorize the task as "conversation_only" or "structured_output_required".
2. For structured output: Point out absent fields, ambiguous data types, unspecified ordering, or missing error-handling.

Do NOT invent issues if unsure. Be conservative in flagging issues.

Output format (strict JSON):
{"has_issues": <bool>, "issues": ["<issue 1>", "<issue 2>"]}"""
            },
            {
                "role": "user",
                "content": f"Analyze this prompt for format issues:\n\n{prompt}"
            }
        ]
    )
    
    result = json.loads(response.choices[0].message.content)
    return Issues(**result)

async def rewrite_prompt(prompt: str, contradiction_issues: List[str], format_issues: List[str]) -> str:
    """Rewrite the prompt to fix identified issues."""
    issues_text = ""
    if contradiction_issues:
        issues_text += "Contradictions found:\n" + "\n".join(f"- {issue}" for issue in contradiction_issues) + "\n\n"
    if format_issues:
        issues_text += "Format issues found:\n" + "\n".join(f"- {issue}" for issue in format_issues) + "\n\n"
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are Dev-Rewriter.

Goal: Rewrite the developer prompt to resolve contradictions and clarify format specifications while preserving the original intent.

Instructions:
1. Fix all identified contradictions
2. Clarify any unclear format requirements
3. Maintain the original purpose and functionality
4. Keep the same level of detail and technical accuracy
5. Preserve the professional tone and engineering context

Return only the rewritten prompt, no additional commentary."""
            },
            {
                "role": "user",
                "content": f"Original prompt:\n{prompt}\n\nIssues to fix:\n{issues_text}\n\nPlease rewrite the prompt to fix these issues:"
            }
        ]
    )
    
    return response.choices[0].message.content

async def optimize_prompt(prompt: str) -> Dict[str, Any]:
    """Main optimization workflow."""
    print("🔍 Analyzing prompt for contradictions...")
    contradiction_result = await check_contradictions(prompt)
    
    print("🔍 Analyzing prompt for format issues...")
    format_result = await check_format_issues(prompt)
    
    print("✏️ Rewriting prompt to fix issues...")
    optimized_prompt = await rewrite_prompt(prompt, contradiction_result.issues, format_result.issues)
    
    return {
        "contradiction_issues": contradiction_result.issues,
        "format_issues": format_result.issues,
        "optimized_prompt": optimized_prompt
    }

async def main():
    """Main function to run the optimization."""
    print("🚀 Starting Prompt Optimization Analysis")
    print("=" * 50)
    
    # Read the original prompt
    with open("prompt_with_process_cheatsheet.txt", "r", encoding="utf-8") as f:
        original_prompt = f.read()
    
    print(f"📄 Original prompt length: {len(original_prompt)} characters")
    print()
    
    # Run optimization
    result = await optimize_prompt(original_prompt)
    
    # Display results
    print("📊 OPTIMIZATION RESULTS")
    print("=" * 50)
    
    if result["contradiction_issues"]:
        print("❌ CONTRADICTIONS FOUND:")
        for issue in result["contradiction_issues"]:
            print(f"  • {issue}")
        print()
    else:
        print("✅ No contradictions found!")
        print()
    
    if result["format_issues"]:
        print("⚠️ FORMAT ISSUES FOUND:")
        for issue in result["format_issues"]:
            print(f"  • {issue}")
        print()
    else:
        print("✅ No format issues found!")
        print()
    
    # Save optimized prompt
    with open("prompt_with_process_cheatsheet_optimized.txt", "w", encoding="utf-8") as f:
        f.write(result["optimized_prompt"])
    
    print("💾 Optimized prompt saved as: prompt_with_process_cheatsheet_optimized.txt")
    print()
    print("📝 OPTIMIZED PROMPT:")
    print("=" * 50)
    print(result["optimized_prompt"])

if __name__ == "__main__":
    asyncio.run(main()) 