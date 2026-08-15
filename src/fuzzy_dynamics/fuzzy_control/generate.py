import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
              
import requests
from datasets import load_dataset
from tqdm import tqdm

# -----------------------------
# QWEN API
# -----------------------------

# process_prompt

def process_prompt(question: str) -> str:
    return (
        f"process_prompt"
    )


def process_calculation_prompt(question: str) -> str:
    return (
        f"process_calculation_prompt"
    )


def call_qwen_api(prompt: str) -> Optional[Dict]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait_s = int(retry_after) if retry_after and retry_after.isdigit() else 10
                print(f"API 429 {wait_s}s  ( {attempt + 1}/{max_retries})")
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"API error ( {attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"API error: {str(e)}")
                return None
    return None


def extract_json_from_text(text: str):
    pattern = r"\{.*?\}"
    matches = re.findall(pattern, text, re.DOTALL)
    for json_str in matches:
        try:
            return json.loads(json_str)
        except Exception:
            continue
    return None


def parse_qwen_response(response: Dict):
    if not response or "choices" not in response or not response["choices"]:
        return None
    content = response["choices"][0]["message"]["content"].strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return extract_json_from_text(content)


def extract_condition_from_text(text: str) -> str:
    pattern = r"(\d+\.?\d*\s*\w+)"
    matches = re.findall(pattern, text)
    return ", ".join(matches)


def extract_answer_from_text(text: str) -> int:
    pattern = r"(\b\d+\b)"
    matches = re.findall(pattern, text)
    if matches:
        return int(matches[-1])
    return 0


def extract_gold_from_answer(answer_text: str) -> int:
    match = re.search(r"####\s*([-+]?\d[\d,]*)", answer_text or "")
    if not match:
        return 0
    num_str = match.group(1).replace(",", "")
    try:
        return int(num_str)
    except ValueError:
        return 0


def normalize_numeric_answer(value):
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        try:
            if "." in text:
                num = float(text)
                return int(num) if num.is_integer() else num
            return int(text)
        except ValueError:
            match = re.search(r"[-+]?\d*\.?\d+", text)
            if match:
                try:
                    num = float(match.group(0))
                    return int(num) if num.is_integer() else num
                except ValueError:
                    return 0
            return 0
    return 0


def sanitize_dataset_name(dataset_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name).strip("_")
    return safe_name or "dataset"


def load_hf_dataset(
    dataset_name: str,
    split: str = "test",
    question_field: Optional[str] = None,
    answer_field: Optional[str] = None,
) -> List[Dict]:
    cache_name = f"{sanitize_dataset_name(dataset_name)}_{split}.json"
    cache_path = Path("data") / cache_name
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    ds = load_dataset(dataset_name, split=split)
    data = []
    for row in ds:
        if question_field:
            question = row.get(question_field, "")
        elif "question_concat" in row:
            question = row.get("question_concat", "")
        elif "Body" in row and "Question" in row:
            body = row.get("Body", "")
            q = row.get("Question", "")
            question = (body + " " + q).strip()
        else:
            question = ""
            for key in ("question", "Question", "problem", "Problem", "prompt", "query", "input", "text"):
                if key in row:
                    question = row.get(key, "")
                    break

        if answer_field:
            raw_answer = row.get(answer_field)
        else:
            raw_answer = None
            for key in ("answer", "Answer", "gold", "label", "target", "output", "solution", "Solution"):
                if key in row:
                    raw_answer = row.get(key)
                    break

        if isinstance(raw_answer, str) and "####" in raw_answer:
            gold = extract_gold_from_answer(raw_answer)
        else:
            gold = normalize_numeric_answer(raw_answer)

        data.append({"question": question, "gold": gold})

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def load_input_file(
    input_file: str,
    hf_split: Optional[str] = None,
    question_field: Optional[str] = None,
    answer_field: Optional[str] = None,
) -> List[Dict]:
    if input_file.startswith("hf:"):
        parts = input_file.split(":", 2)
        dataset_name = parts[1] if len(parts) > 1 else ""
        split = parts[2] if len(parts) > 2 and parts[2] else (hf_split or "test")
        if not dataset_name:
            raise ValueError("HF dataset name is empty.")
        return load_hf_dataset(
            dataset_name,
            split=split,
            question_field=question_field,
            answer_field=answer_field,
        )
    path = Path(input_file)
    if path.exists():
        if path.suffix.lower() == ".jsonl":
            data = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    data.append(obj)
            return data
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix.lower() in {".json", ".jsonl"}:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if hf_split is not None or "/" in input_file:
        return load_hf_dataset(
            input_file,
            split=hf_split or "test",
            question_field=question_field,
            answer_field=answer_field,
        )
    raise FileNotFoundError(f"Input file not found: {input_file}")


def process_file(
    input_file: str,
    output_file: str,
    hf_split: Optional[str] = None,
    question_field: Optional[str] = None,
    answer_field: Optional[str] = None,
):
    data = load_input_file(
        input_file,
        hf_split=hf_split,
        question_field=question_field,
        answer_field=answer_field,
    )
    error_count = 0
    skipped = []
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write("[")
        first = True
        for idx, item in enumerate(tqdm(data, desc="Processing"), start=1):
            if not isinstance(item, dict) or "question" not in item:
                error_count += 1
                skipped.append(idx)
                continue

            question = item.get("question", "")
            gold = item.get("gold", 0)
            prompt = process_prompt(question)
            try:
                response = call_qwen_api(prompt)
                result = parse_qwen_response(response) if response else None
                if result:
                    operation_condition = result.get("operation condition", "") or extract_condition_from_text(question)
                    calculation_process = result.get("calculation process", "")
                    answer = int(result.get("answer", 0)) or extract_answer_from_text(question)

                    if not calculation_process:
                        second_prompt = process_calculation_prompt(question)
                        second_response = call_qwen_api(second_prompt)
                        second_result = parse_qwen_response(second_response) if second_response else None
                        if second_result:
                            calculation_process = second_result.get("calculation process", "")
                else:
                    operation_condition = extract_condition_from_text(question)
                    calculation_process = ""
                    answer = extract_answer_from_text(question)

                processed_item = {
                    "question": question,
                    "gold": gold,
                    "operation condition": operation_condition,
                    "calculation process": calculation_process,
                    "answer": answer,
                }
            except Exception as e:
                error_count += 1
                print(f"error")
                processed_item = {
                    "question": question,
                    "gold": gold,
                    "operation condition": extract_condition_from_text(question),
                    "calculation process": f"API error: {str(e)}",
                    "answer": extract_answer_from_text(question),
                }

            if not first:
                f_out.write(",\n")
            json.dump(processed_item, f_out, ensure_ascii=False)
            f_out.flush()
            first = False
            time.sleep(0.5)  # 避免API速率限制

        f_out.write("]\n")

    print(f"\n{output_file}")
    if error_count > 0:
        print(f"{error_count}: {skipped}")


if __name__ == "__main__":
    DATASET_NAME = ""
    DATASET_SPLIT = ""
    QUESTION_FIELD = None
    ANSWER_FIELD = None

    INPUT_FILE = DATASET_NAME
    OUTPUT_FILE = ""
    process_file(
        INPUT_FILE,
        OUTPUT_FILE,
        hf_split=DATASET_SPLIT,
        question_field=QUESTION_FIELD,
        answer_field=ANSWER_FIELD,
    )
