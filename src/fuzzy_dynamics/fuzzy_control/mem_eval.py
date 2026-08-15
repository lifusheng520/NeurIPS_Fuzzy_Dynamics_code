import json
import re

from .generate import call_qwen_api, parse_qwen_response

INPUT_JSON_PATH = r""

def call_llm(prompt):
    response = call_qwen_api(prompt)
    if not response:
        return "Error: empty response from Qwen API"
    parsed = parse_qwen_response(response)
    if isinstance(parsed, dict):
        return json.dumps(parsed, ensure_ascii=False)
    if parsed is None:
        return "Error: unable to parse Qwen response"
    return str(parsed)

def extract_json_from_text(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                return json.loads(json_str)
        except:
            return {"error": "Failed to extract JSON from response", "raw_response": text}
    
    return {"error": "No JSON found in response", "raw_response": text}

def evaluate_operational_accuracy(question, calculation_process, membership_value):
    prompt = f""""""
    response = call_llm(prompt)
    return extract_json_from_text(response)

def evaluate_information_extraction(question, operation_condition, membership_value):
    prompt = f""""""
    response = call_llm(prompt)
    return extract_json_from_text(response)

def evaluate_intermediate_step(question, calculation_process, membership_value):
    prompt = f""""""
    response = call_llm(prompt)
    return extract_json_from_text(response)

def evaluate_answer_conformity(question, answer, membership_value):
    prompt = f""""""
    response = call_llm(prompt)
    return extract_json_from_text(response)

def load_input_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f" '{file_path}' ")
        return None
    except json.JSONDecodeError:
        print(f" '{file_path}' ")
        return None

def evaluate_single_item(item, index):
    question = item["question"]
    operation_condition = item["operation condition"]
    calculation_process = item["calculation process"]
    answer = item["answer"]
    membership_values = item["membership"]
    
    print(f"\n{'='*60}")
    print(f"{index + 1} ")
    print(f"{'='*60}")
    
    result_oa = evaluate_operational_accuracy(question, calculation_process, membership_values['Operational Accuracy'])
    print(json.dumps(result_oa, indent=2))
    
    result_ie = evaluate_information_extraction(question, operation_condition, membership_values['Information Extraction Fidelity'])
    print(json.dumps(result_ie, indent=2))
    
    result_is = evaluate_intermediate_step(question, calculation_process, membership_values['Intermediate Step Validity'])
    print(json.dumps(result_is, indent=2))

    result_ac = evaluate_answer_conformity(question, answer, membership_values['Answer Conformity'])
    print(json.dumps(result_ac, indent=2))
    
    print(f"\n{'='*60}")
    print(f" {index + 1}:")
    print(f"{'='*60}")
    
    results = [
        ("", result_oa),
        ("", result_ie),
        ("", result_is),
        ("", result_ac)
    ]
    
    for name, result in results:
        if "error" in result:
            print(f"{name} - {result['error']}")
            if "raw_response" in result:
                print(f": {result['raw_response'][:200]}...")  
        else:
            supported = result.get("supported", False)
            print(f"{name}: {'yes' if supported else 'no'}")
    
    return {
        "question": question,
        "evaluation_results": {
            "Operational Accuracy": result_oa,
            "Information Extraction Fidelity": result_ie,
            "Intermediate Step Validity": result_is,
            "Answer Conformity": result_ac
        }
    }

def main():
    input_data = load_input_data(INPUT_JSON_PATH)
    
    if input_data is None:
        return
    
    if not isinstance(input_data, list):
        input_data = [input_data]
    
    all_results = []
    
    for i, item in enumerate(input_data):
        result = evaluate_single_item(item, i)
        all_results.append(result)
    
    output_path = INPUT_JSON_PATH.replace(".json", "")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n: {output_path}")

if __name__ == "__main__":
    main()
