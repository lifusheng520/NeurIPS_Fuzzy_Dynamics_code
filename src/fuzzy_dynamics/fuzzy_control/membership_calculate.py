import json
import time
import re
from tqdm import tqdm

from .generate import call_qwen_api, parse_qwen_response

def create_evaluation_prompt(item):
    """
    """
    prompt = f""""""
    return prompt

def call_llm_api(prompt):
    """
    Use the shared Qwen API caller from generate.py to get a parsed JSON response.
    """
    max_retries = 3
    for attempt in range(max_retries):
        response = call_qwen_api(prompt)
        if not response:
            if attempt < max_retries - 1:
                print(f"API error ( {attempt + 1}/{max_retries})")
                time.sleep(2)
                continue
            return None
        parsed = parse_qwen_response(response)
        if parsed:
            return parsed
        if attempt < max_retries - 1:
            print(f"error ( {attempt + 1}/{max_retries})")
            time.sleep(2)
    return None

def extract_json_from_text(text):
    pattern = r'\{[\s\S]*\}'
    matches = re.findall(pattern, text, re.DOTALL)
    for json_str in matches:
        try:
            json_str = json_str.replace('```json', '').replace('```', '')
            return json.loads(json_str)
        except json.JSONDecodeError:
            continue
    return None

def parse_llm_response(response):
    if not response:
        return None
    if isinstance(response, dict):
        return response
    return extract_json_from_text(str(response))

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    error_count = 0
    
    for item in tqdm(data, desc="Evaluating"):
        try:
            prompt = create_evaluation_prompt(item)

            response = call_llm_api(prompt)

            result = parse_llm_response(response) if response else None
            
            if result and 'membership' in result:
                item['membership'] = result['membership']
            else:
                error_count += 1
                print(f": {item['question']}")
                item['membership'] = {
                    "Operational Accuracy": 0.0,
                    "Information Extraction Fidelity": 0.0,
                    "Intermediate Step Validity": 0.0,
                    "Answer Conformity": 0.0
                }
            
            processed_data.append(item)
            time.sleep(1)
            
        except Exception as e:
            error_count += 1
            print(f" '{item['question']}': {str(e)}")
            item['membership'] = {
                "Operational Accuracy": 0.0,
                "Information Extraction Fidelity": 0.0,
                "Intermediate Step Validity": 0.0,
                "Answer Conformity": 0.0
            }
            processed_data.append(item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{output_file}")
    if error_count > 0:
        print(f"error_count: {error_count}")

if __name__ == "__main__":
    INPUT_FILE = r""
    OUTPUT_FILE = ""
    process_file(INPUT_FILE, OUTPUT_FILE)
