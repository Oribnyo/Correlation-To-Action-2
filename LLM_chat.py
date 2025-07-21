import openai
import sys
import os
from dotenv import load_dotenv
from LLM_report_theme import save_llm_response_to_word
import pdfplumber
import faiss
import numpy as np
import pickle

# Load environment variables from .env file
load_dotenv()

client = openai.OpenAI()

# Load system prompt
with open("prompt_with_process_cheatsheet_5whys.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

class RAGLibrary:
    def __init__(self, pdf_path, index_path="rag_index_openai.pkl", chunk_size=500):
        self.pdf_path = pdf_path
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = None
        self.index = None
        self._load_or_build_index()

    def _extract_text_chunks(self):
        with pdfplumber.open(self.pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i+self.chunk_size])
            chunks.append(chunk)
        return chunks

    def _get_openai_embeddings(self, texts):
        # Use OpenAI's embedding API (text-embedding-3-small or ada-002)
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [np.array(d.embedding, dtype=np.float32) for d in response.data]

    def _build_index(self):
        self.chunks = self._extract_text_chunks()
        self.embeddings = self._get_openai_embeddings(self.chunks)
        self.index = faiss.IndexFlatL2(len(self.embeddings[0]))
        self.index.add(np.vstack(self.embeddings))
        with open(self.index_path, "wb") as f:
            pickle.dump((self.chunks, self.embeddings), f)

    def _load_or_build_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                self.chunks, self.embeddings = pickle.load(f)
            self.index = faiss.IndexFlatL2(len(self.embeddings[0]))
            self.index.add(np.vstack(self.embeddings))
        else:
            self._build_index()

    def retrieve(self, query, top_k=2):
        query_emb = self._get_openai_embeddings([query])[0].reshape(1, -1)
        D, I = self.index.search(query_emb, top_k)
        return [self.chunks[i] for i in I[0]]

# Usage example:
# rag = RAGLibrary("/Users/oribenyosef/Correlation-To-Action-2/domain_specific_library/five-whys-technique.pdf")
# print(rag.retrieve("What is the Five Whys technique?"))

def load_csv_data(csv_file_path):
    """Load and return CSV content with timestamp conversion and column renaming"""
    if not os.path.exists(csv_file_path):
        print(f"❌ Error: CSV file '{csv_file_path}' not found!")
        return None
    
    try:
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Rename columns with proper sensor descriptions
        column_mapping = {
            'timestamp': 'timestamp',
            'Sensor 1': 'Sensor 1 [Hydrocarbon_Dew_Point_C]',
            'Sensor 2': 'Sensor 2 [Train_1_Gas_FlowRate_MMBTU_HR]',
            'Sensor 3': 'Sensor 3 [Train_1_Gas_Temperature_C]',
            'Sensor 4': 'Sensor 4 [Train_1_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 5': 'Sensor 5 [Train_1_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 6': 'Sensor 6 [Train_2_Gas_FlowRate_MMBTU_HR]',
            'Sensor 7': 'Sensor 7 [Train_2_Gas_Temperature_C]',
            'Sensor 8': 'Sensor 8 [Train_2_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 9': 'Sensor 9 [Train_2_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 10': 'Sensor 10 [Train_3_Gas_FlowRate_MMBTU_HR]',
            'Sensor 11': 'Sensor 11 [Train_3_Gas_Temperature_C]',
            'Sensor 12': 'Sensor 12 [Train_3_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 13': 'Sensor 13 [Train_3_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 14': 'Sensor 14 [Train_4_Gas_FlowRate_MMBTU_HR]',
            'Sensor 15': 'Sensor 15 [Train_4_Gas_Temperature_C]',
            'Sensor 16': 'Sensor 16 [Train_4_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 17': 'Sensor 17 [Train_4_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 18': 'Sensor 18 [Train_5_Gas_FlowRate_MMBTU_HR]',
            'Sensor 19': 'Sensor 19 [Train_5_Gas_Temperature_C]',
            'Sensor 20': 'Sensor 20 [Train_5_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 21': 'Sensor 21 [Train_5_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 22': 'Sensor 22 [Train_6_Gas_FlowRate_MMBTU_HR]',
            'Sensor 23': 'Sensor 23 [Train_6_Gas_Temperature_C]',
            'Sensor 24': 'Sensor 24 [Train_6_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 25': 'Sensor 25 [Train_6_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 26': 'Sensor 26 [Train_7_Gas_FlowRate_MMBTU_HR]',
            'Sensor 27': 'Sensor 27 [Train_7_Gas_Temperature_C]',
            'Sensor 28': 'Sensor 28 [Train_7_Liquid_Out_1_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 29': 'Sensor 29 [Train_7_Liquid_Out_2_Gallons_Day_Cumulative_zeros_every_24h]',
            'Sensor 30': 'Sensor 30 [Entrance_Pressure_psig]',
            'Sensor 31': 'Sensor 31 [Delivery_Pressure_psig]'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Create output directory if it doesn't exist
        output_dir = "snapshots for llm"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the processed CSV file in "snapshots for llm" folder
        base_filename = os.path.basename(csv_file_path)
        output_filename = f"{output_dir}/{base_filename}"
        df.to_csv(output_filename, index=False)
        
        # Convert back to CSV string for LLM
        csv_content = df.to_csv(index=False)
        
        print(f"✅ Loaded and processed CSV file: {csv_file_path}")
        print(f"✅ Renamed columns for clarity")
        print(f"✅ Saved processed CSV as: {output_filename}")
        return csv_content
        
    except Exception as e:
        print(f"❌ Error reading CSV file: {e}")
        return None

def chat_with_llm(csv_content, rag):
    """Interactive chat with LLM about CSV data"""
    print("\n🤖 LLM Chat Session Started")
    print("💬 You can now chat with the LLM about the CSV data")
    print("📝 Type 'quit' or 'exit' to end the conversation")
    print("=" * 50)
    
    # Initialize conversation with CSV data
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""
Hello Senior Process Engineer,
I am the facility operator and I need your expertise. Our deep learning model has forecasted a Hydrocarbon Dew Point (HCDP) threshold violation within the next 30 minutes. You are provided with the forecasted 30-minute multivariate time-series data and 96-minutes windows preciding the HCDP violation in the attached CSV file.

CSV DATA:
{csv_content}

Please provide the following outputs:
1. Flowrate Adjustment Plan:
   - Determine the minimum reduction in total flowrate (across Sensor 2 [Train_1_Gas_FlowRate_MMBTU_HR], Sensor 6 [Train_2_Gas_FlowRate_MMBTU_HR], Sensor 10 [Train_3_Gas_FlowRate_MMBTU_HR], Sensor 14 [Train_4_Gas_FlowRate_MMBTU_HR], Sensor 18 [Train_5_Gas_FlowRate_MMBTU_HR], Sensor 22 [Train_6_Gas_FlowRate_MMBTU_HR], Sensor 26 [Train_7_Gas_FlowRate_MMBTU_HR]) required to prevent HCDP from exceeding 5.40 °C.
   - Specify the start and end time for the reduction window.
   - Explain your reasoning and calculation steps clearly.

2. Root Cause Analysis:
   - Use the Five Whys technique to identify the most probable root cause for the HCDP violation.
   - Provide a short explanation of your reasoning.

3. Corrective Actions
   - Based on the identified root cause and your professional knowledge, suggest realistic and relevant corrective actions to prevent future occurrences.

Assume you are a senior process engineer familiar with the system, and the natural gas processing domain.

Remember to utilize your knowledge of all 31 sensors in the system, including:
- Flow rates from all 7 trains
- Temperature profiles
- Liquid removal rates
- System pressures

Use the complete process understanding from the system prompt.
"""}
    ]   
    
    # Get initial response
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-2025-04-14",
            messages=messages,
            temperature=0.1
        )
        llm_response = response.choices[0].message.content
        print(f"\n🤖 LLM: {llm_response}")
        
        # Save response to Word document
        try:
            report_filename = save_llm_response_to_word(llm_response, "prompt_with_process_cheatsheet_5whys.txt")
            print(f"📄 Report saved: {report_filename}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save Word report: {e}")
        
        messages.append({"role": "assistant", "content": llm_response})
    except Exception as e:
        print(f"❌ Error getting initial response: {e}")
        return
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Chat session ended. Goodbye!")
                break
            elif user_input.lower() in ['clear', 'reset', 'refresh']:
                print("🔄 Clearing conversation history...")
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""
Hello Senior Process Engineer,
I am the facility operator and I need your expertise. Our deep learning model has forecasted a Hydrocarbon Dew Point (HCDP) threshold violation within the next 30 minutes. You are provided with the forecasted 30-minute multivariate time-series data and 96-minutes windows preciding the HCDP violation in the attached CSV file.

CSV DATA:
{csv_content}

Please provide the following outputs:
1. Flowrate Adjustment Plan:
   - Determine the minimum reduction in total flowrate (across Sensor 2 [Train_1_Gas_FlowRate_MMBTU_HR], Sensor 6 [Train_2_Gas_FlowRate_MMBTU_HR], Sensor 10 [Train_3_Gas_FlowRate_MMBTU_HR], Sensor 14 [Train_4_Gas_FlowRate_MMBTU_HR], Sensor 18 [Train_5_Gas_FlowRate_MMBTU_HR], Sensor 22 [Train_6_Gas_FlowRate_MMBTU_HR], Sensor 26 [Train_7_Gas_FlowRate_MMBTU_HR]) required to prevent HCDP from exceeding 5.40 °C.
   - Specify the start and end time for the reduction window.
   - Explain your reasoning and calculation steps clearly.

2. Root Cause Analysis:
   - Use the Five Whys technique to identify the most probable root cause for the HCDP violation.
   - Provide a short explanation of your reasoning.

3. Corrective Actions
   - Based on the identified root cause and your professional knowledge, suggest realistic and relevant corrective actions to prevent future occurrences.

Assume you are a senior process engineer familiar with the system, and the natural gas processing domain.

Remember to utilize your knowledge of all 31 sensors in the system, including:
- Flow rates from all 7 trains
- Temperature profiles
- Liquid removal rates
- System pressures

Use the complete process understanding from the system prompt.
"""}
                ]
                print("✅ Conversation cleared! Starting fresh.")
                continue
            
            if not user_input:
                continue
            
            # RAG: Retrieve context for the user query
            rag_context = rag.retrieve(user_input, top_k=2)
            context_text = "\n\n".join(rag_context)
            # Add context to the user message
            user_message = f"{user_input}\n\n[Relevant Reference Material: Five Whys Technique]\n{context_text}"

            messages.append({"role": "user", "content": user_message})
            
            # Get LLM response
            response = client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=messages,
                temperature=0.1
            )
            
            llm_response = response.choices[0].message.content
            print(f"\n🤖 LLM: {llm_response}")
            
            # Add LLM response to conversation
            messages.append({"role": "assistant", "content": llm_response})
            
        except KeyboardInterrupt:
            print("\n👋 Chat session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

def main():
    # Get CSV file path from command line argument or use default
    csv_file_path = sys.argv[1] if len(sys.argv) > 1 else "data/snapshot for llm/19-6-25.csv"
    
    # Load CSV data
    csv_content = load_csv_data(csv_file_path)
    if csv_content is None:
        print("Usage: python LLM_chat.py <path_to_csv_file>")
        sys.exit(1)
    
    # Initialize RAG library
    rag = RAGLibrary("domain_specific_library/five-whys-technique.pdf")

    # Start interactive chat
    chat_with_llm(csv_content, rag)

if __name__ == "__main__":
    main()