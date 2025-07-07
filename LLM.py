from openai import OpenAI
client = OpenAI()

with open("prompt.txt", "r", encoding="utf-8") as f:
    instructions = f.read()

user_input = """
Hello Senior Process Engineer,
I am the facility operator and I need your expertise. Our deep learning model has forecasted a Hydrocarbon Dew Point (HCDP) threshold violation within the next 30 minutes. You are provided with the forecasted 30-minute multivariate time-series data in the attached CSV file.
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

response = client.responses.create(
    model="gpt-4.1-2025-04-14",
    instructions=instructions,
    input="How would I declare a variable for a last name?",
)

print(response.output_text)
