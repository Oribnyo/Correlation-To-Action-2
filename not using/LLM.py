from openai import OpenAI
import sys
import os

client = OpenAI()

with open("prompt.txt", "r", encoding="utf-8") as f:
    instructions = f.read()

# Get CSV file path from command line argument or use default
csv_file_path = sys.argv[1] if len(sys.argv) > 1 else "snapshots/snapshot_438_hcdp_8.7.csv"

# Check if file exists
if not os.path.exists(csv_file_path):
    print(f"❌ Error: CSV file '{csv_file_path}' not found!")
    print("Usage: python LLM.py <path_to_csv_file>")
    print("Example: python LLM.py snapshots/snapshot_001_hcdp_10.9.csv")
    sys.exit(1)

# Read CSV file content
try:
    with open(csv_file_path, "r", encoding="utf-8") as f:
        csv_content = f.read()
    print(f"✅ Loaded CSV file: {csv_file_path}")
except Exception as e:
    print(f"❌ Error reading CSV file: {e}")
    sys.exit(1)

user_input = f"""
Hello Senior Process Engineer,
I am the facility operator and I need your expertise. Our deep learning model has forecasted a Hydrocarbon Dew Point (HCDP) threshold violation within the next 30 minutes. You are provided with the forecasted 30-minute multivariate time-series data in the attached CSV file.

CSV DATA:
{csv_content}

Please provide the following outputs:
1. Flowrate Adjustment Plan:
   - Determine the minimum reduction in total flowrate (across sensors 2, 6, 10, 14, 18, 22, 26) required to prevent HCDP from exceeding 5°C.
   - Specify the start and end time for the reduction window.
   - Explain your reasoning and calculation steps clearly.

2. Root Cause Analysis:
   - Use the Five Whys technique to identify the most probable root cause for the HCDP violation.
   - Provide a short explanation of your reasoning.

3. Corrective Actions
   - Based on the identified root cause and your professional knowledge, suggest realistic and relevant corrective actions to prevent future occurrences.

Assume you are a senior process engineer familiar with the system, and the natural gas processing domain.
Please return all answers in a structured JSON format with keys: `flowrate_plan`, `root_cause_analysis`, and `corrective_actions`.
"""

response = client.chat.completions.create(
    model="gpt-4.1-2025-04-14",
    messages=[
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_input}
    ],
    temperature=0.1
)

print(response.choices[0].message.content)
