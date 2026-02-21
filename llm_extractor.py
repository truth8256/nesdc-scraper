
import os
import json
import google.generativeai as genai
from typing import List, Dict, Any

# Configure API Key
# Expects OPENAI_API_KEY in environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print(f"[INFO] OpenAI API Key found. Using model: gpt-4o-mini")
else:
    openai_client = None
    print("[WARN] Warning: No OPENAI_API_KEY found.")

# Configure Gemini API Key
# Expects GEMINI_API_KEY or GOOGLE_API_KEY in environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("[WARN] Warning: No API Key found for Gemini (GEMINI_API_KEY or GOOGLE_API_KEY).")

def extract_table_from_markdown(markdown_text: str, context: str = "") -> List[Dict[str, Any]]:
    """
    Extracts survey table data from markdown text using OpenAI (Priority) or Gemini (Fallback).
    Returns a list of dictionaries compatible with the project's JSON schema.
    """
    
    # 1. Try OpenAI first
    if openai_client:
        return extract_with_openai(markdown_text, context)
    
    # 2. Fallback to Gemini
    if GEMINI_API_KEY:
        return extract_with_gemini(markdown_text, context)

    print("[ERROR] No valid API keys found for LLM extraction.")
    return []

def extract_with_openai(markdown_text: str, context: str) -> List[Dict[str, Any]]:
    try:
        prompt = _build_prompt(markdown_text, context)
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data extraction assistant. extracting survey data from OCR'd text. Return only pure JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        text_response = response.choices[0].message.content.strip()
        return _parse_json_response(text_response)

    except Exception as e:
        print(f"[Error] OpenAI Extraction Failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def extract_with_gemini(markdown_text: str, context: str) -> List[Dict[str, Any]]:
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = _build_prompt(markdown_text, context)
        
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        return _parse_json_response(text_response)
    
    except Exception as e:
        print(f"[Error] Gemini Extraction Failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def _build_prompt(markdown_text, context):
    return f"""
    Context: {context}
    
    Input Text (Markdown/OCR):
    {markdown_text}
    
    Goal: Identify the survey data table and extract ALL sub-tables (Demographics) into a single structured JSON format.
    
    The JSON structure must be:
    {{
        "question": "The survey question text or table title found above the table",
        "data": [
            {{
                "group": "The demographic group (e.g., 전체, 성별, 연령, 권역, 지지정당 etc.)",
                "category": "The specific item within the group (e.g., 남성, 여성, 30대, 서울, 국민의힘 etc.)",
                "n": "Case count (integer). If multiple columns exist (e.g., '조사 완료', '가중값 적용'), use 'n_raw' for actual and 'n_weighted' for weighted, or include both in the object.",
                "responses": {{
                    "Respose Option 1": 40.5,
                    "Response Option 2": 30.2,
                    ...
                }}
            }},
            ...
        ]
    }}

    IMPORTANT: 
    1. READ the text above the table to find the "question" or "title".
    2. The survey results are broken down by demographic groups. You MUST extract ALL of them.
    3. You must infer the "group" name based on the category. 
       - If category is "남성", "여성" -> group is "성별"
       - If category is "18-29세", "30대" -> group is "연령"
       - If category is "서울", "인천/경기" -> group is "권역"
       - If category is "전체" or "Total" -> group is "전체" (or "Total")
    4. **Case Counts**: Often there are two columns for "n": "조사 완료 사례수" (Actual) and "가중값 적용 사례수" (Weighted).
       - If both exist, keys MUST be "n_raw" and "n_weighted".
       - If only one exists, use "n".

    Structure Example:
    {{
        "question": "Q1. 질문 내용...",
        "data": [
            {{ "group": "전체", "category": "전체", "n_raw": 1000, "n_weighted": 1000, "responses": {{ ... }} }},
            {{ "group": "성별", "category": "남성", "n_raw": 490, "n_weighted": 495, "responses": {{ ... }} }},
            ...
        ]
    }}

    Rules:
    1. Extract ONLY the JSON. No markdown formatting.
    2. Convert percentages to numbers.
    3. "n", "n_raw", "n_weighted" must be numbers.
    4. "responses" object should contain the percentage values for each response option.
    5. Handle merged cells intelligently.
    """

def _parse_json_response(text_response):
    # Cleanup code blocks if present
    if text_response.startswith("```json"):
        text_response = text_response[7:]
    elif text_response.startswith("```"):
         text_response = text_response[3:]
         
    if text_response.endswith("```"):
        text_response = text_response[:-3]
        
    return json.loads(text_response.strip())

if __name__ == "__main__":
    # Test stub
    pass
