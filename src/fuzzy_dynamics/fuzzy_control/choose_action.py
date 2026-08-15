import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import os
from typing import List, Dict, Tuple, Any

from .generate import call_qwen_api, parse_qwen_response

# Helper to safely handle list/dict responses from LLM
def _first_dict(obj: Any) -> Dict:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj[0]
    return {}

MEM_EVAL_ENABLED = False
MEM_EVAL_PATH = ""
# ==================== CONSTANT DEFINITIONS (RENAMED) ====================
# Renamed from STATE_KEYS_ORDERED to reflect direct use of membership values.
MEMBERSHIP_KEYS_ORDERED = [
    "Operational Accuracy",
    "Information Extraction Fidelity",
    "Intermediate Step Validity",
    "Answer Conformity"
]

# ==================== API ACTION EXECUTOR ====================
class ActionExecutor:
    def __init__(self, api_key: str, api_base: str = ""):
        # api_key/api_base kept for backward compatibility; call_qwen_api uses generate.py configuration.
        self.api_key = api_key
        self.api_base = api_base
        self.headers = {
            'Content-Type': '',
        }
    
    @staticmethod
    def clean_and_extract_json(response_text: str) -> str:
        try:
            json.loads(response_text)
            return response_text 
        except json.JSONDecodeError:
            pass
        
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return response_text[start_idx:end_idx+1]
        
        return response_text


    def call_gemini_api(self, prompt: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            try:
                response = call_qwen_api(prompt)
                if not response:
                    raise ValueError("empty")
                parsed = parse_qwen_response(response)
                if isinstance(parsed, (dict, list)):
                    return json.dumps(parsed, ensure_ascii=False)
                if isinstance(parsed, str):
                    return parsed
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content
            except Exception as e:
                print(f" {attempt+1} : {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2) 
                else:
                    raise Exception(f": {str(e)}")
    
    def break_into_subproblems(self, question: str) -> str:
        prompt = f""""""
        return self.call_gemini_api(prompt)
    
    def identify_error_and_step(self, question: str, calculation_process: str = "") -> str:
        
        prompt = f""""""
        return self.call_gemini_api(prompt)
    
    def extract_entities_and_triples(self, question: str) -> str:
        prompt = f""""""
        return self.call_gemini_api(prompt)
    
    def provide_reasoning_chain_example(self, question: str) -> str:
        prompt = f""""""
        return self.call_gemini_api(prompt)


    # ============membership calculate  ============
    def evaluate_membership(self, question: str, operation_condition: str, 
                           calculation_process: str, answer: str) -> str:
        prompt = f""""""
        return self.call_gemini_api(prompt)



MEMBERSHIP_LINGUISTIC_TERMS = {
    "Operational Accuracy": "Accurate",
    "Information Extraction Fidelity": "Correct",
    "Intermediate Step Validity": "Accurate",
    "Answer Conformity": "Reasonable"
}

# ==================== HELPER FUNCTION (RENAMED) ====================
def get_membership_vector(membership_dict: Dict) -> List[float]:
    """
    Safely creates a membership vector (list of floats) from a membership dictionary.
    This function handles both nested ({'State': {'Term': 0.9}}) and
    flat ({'State': 0.9}) structures for robustness.
    Renamed from get_state_list_from_membership.
    """
    membership_list = []
    if not isinstance(membership_dict, dict):
        return [0.0] * len(MEMBERSHIP_KEYS_ORDERED)
        
    for k in MEMBERSHIP_KEYS_ORDERED:
        value = membership_dict.get(k)
        score = 0.0
        if isinstance(value, dict):
            
            score = value.get(MEMBERSHIP_LINGUISTIC_TERMS[k], 0.0)
        elif isinstance(value, (int, float)):
            
            score = float(value)
        membership_list.append(score)
    return membership_list

# ==================== REWARD AND DATA PARSING (MODIFIED) ====================

ALPHA_1 = 1.0   
ALPHA_2 = 0.1   
ALPHA_3 = 0.0 

def compute_reward(result_entry_after_action: Dict) -> float:
    """
        R = α1 * ((min_i μ_i)^2 - 1)  - α2 * N  + α3 * log det(B B^T)
    """
    membership_values = result_entry_after_action.get("membership", {})
    if not isinstance(membership_values, dict) or not membership_values:
        return -1.0  

    mu_list = []
    for key in MEMBERSHIP_KEYS_ORDERED:
        try:
            mu = float(membership_values.get(key, 0.0))
        except (TypeError, ValueError):
            mu = 0.0
        mu_list.append(mu)

    min_mu = min(mu_list) if mu_list else 0.0
    accuracy_term = (min_mu ** 2) - 1.0    
    accuracy_term *= ALPHA_1

    N = result_entry_after_action.get("num_iterations", 1)
    try:
        N = max(1, int(N))
    except (TypeError, ValueError):
        N = 1
    efficiency_term = - ALPHA_2 * N

    diversity_term = 0.0  

    reward = accuracy_term + efficiency_term + diversity_term

    reward = max(-1.0, min(1.0, reward))
    return reward

class KnowledgeController(nn.Module):
    def __init__(self, rules: List[Tuple[Dict[str, str], str]]):
        super().__init__()
        self.rules = rules
        self.rule_parameters = nn.ParameterList()
        for num_conditions, _ in self.rules:
            # Match number of conditions to the number of membership keys.
            num_conditions = len(MEMBERSHIP_KEYS_ORDERED) 
            # +1 for the overall rule weight
            rule_param = nn.Parameter(torch.randn(num_conditions + 1))
            self.rule_parameters.append(rule_param)

    def compute_rule_strength(self, membership_dict: Dict[str, Dict[str, float]]) -> Tuple[List[Tuple[torch.Tensor, str]], Dict[str, Any]]:
        """
        Computes rule strength directly from the input membership dictionary.
        This function is robust and handles both flat and nested membership data.
        """
        strengths = []
        for i, (conditions, action) in enumerate(self.rules):
            rule_params = self.rule_parameters[i]
            softplus = nn.Softplus()
            exp_rule_params = softplus(rule_params)
            
            if not conditions:
                rule_strength = exp_rule_params[-1]
            else:
                condition_weights = exp_rule_params[:-1]
                rule_overall_weight = exp_rule_params[-1]
                
                membership_scores = []
                # Iterate through membership variables in a fixed order to match weights.
                for var_name in MEMBERSHIP_KEYS_ORDERED:
                    linguistic_term = MEMBERSHIP_LINGUISTIC_TERMS[var_name]
                    
                    # Safely handle flat and nested membership data.
                    value = membership_dict.get(var_name)
                    term_membership = 0.0
                    if isinstance(value, dict):
                        term_membership = value.get(linguistic_term, 0.0)
                    elif isinstance(value, (int, float)):
                        term_membership = float(value)
                    
                    # Apply 'IS' (True) or 'IS NOT' (False) logic from rule definition.
                    is_condition = conditions.get(var_name, True) # Default to 'IS'
                    score = term_membership if is_condition else (1.0 - term_membership)
                    membership_scores.append(score)
                
                cond_tensor = torch.tensor(membership_scores, device=rule_params.device, dtype=rule_params.dtype)
                weighted = condition_weights * cond_tensor
                min_score = torch.min(weighted)  # Fuzzy 'AND'
                rule_strength = rule_overall_weight * min_score
            
            strengths.append((rule_strength, action))
        
        return strengths, membership_dict


    def get_rule_weights_info(self) -> str:
        softplus = nn.Softplus()
        info_lines = ["\n" + "="*50, "rule:", "="*50]
        for i, ((conditions, action), params) in enumerate(zip(self.rules, self.rule_parameters)):
            exp_params = softplus(params)
            condition_weights = exp_params[:-1].detach().cpu().numpy()
            rule_weight = exp_params[-1].item()
            
            cond_desc_parts = []
            for var, is_cond in conditions.items():
                op = "yes" if is_cond else "no"
                term = MEMBERSHIP_LINGUISTIC_TERMS[var]
                cond_desc_parts.append(f"{var} {op} {term}")

            cond_desc = " and ".join(cond_desc_parts)
            info_lines.append(f"rule {i+1}: [condition] {cond_desc}")
            info_lines.append(f"    action: {action}")
            info_lines.append(f"    rule weight: {np.round(condition_weights, 4)}")
            info_lines.append(f"    : {rule_weight:.4f}")
            info_lines.append("-"*50)
        return "\n".join(info_lines)

    def get_action_preference(self, membership_dict: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """Takes membership_dict as input."""
        rule_strengths_with_actions, _ = self.compute_rule_strength(membership_dict)
        action_preferences = {}
        unique_actions = list(dict.fromkeys([action for _, action in self.rules]))

        for action_name in unique_actions:
            if action_name not in action_preferences:
                device = self.rule_parameters[0].device
                action_preferences[action_name] = torch.tensor(0.0, device=device)
        
        for strength, action_name in rule_strengths_with_actions:
            action_preferences[action_name] = torch.max(action_preferences[action_name], strength)
        
        return action_preferences

class KoGuN(nn.Module):
    
    def __init__(self, rules: List[Tuple[Dict[str, bool], str]], membership_dim: int, action_dim: int, temperature: float = 0.8, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.membership_dim = membership_dim    
        self.action_dim = action_dim            
        

        self.knowledge_controller = KnowledgeController(rules).to(self.device)
        

        self.policy_network = nn.Sequential(
            nn.Linear(membership_dim, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim)
        ).to(self.device)
        

        self.value_network = nn.Sequential(
            nn.Linear(membership_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ).to(self.device)
        
        self.temperature = temperature
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        
        self.action_names = list(dict.fromkeys([rule[1] for rule in rules]))
        self.log_file = None

    def set_log_file(self, log_file: str):
        self.log_file = log_file

    def forward(self, membership_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:


        scaled_vector = torch.clamp(membership_vector, 0.0, 1.0)


        if scaled_vector.dim() == 1:
            scaled_vector = scaled_vector.unsqueeze(0)


        logits = self.policy_network(scaled_vector)           # [B, action_dim]
        action_probs = torch.softmax(
            logits / max(0.1, self.temperature), dim=-1
        )                                                     # [B, action_dim]

        state_values = self.value_network(scaled_vector)      # [B, 1]

        return action_probs, state_values

    def get_action(self, membership_dict: Dict[str, float]) -> Tuple[int, Dict]:
        """Takes membership_dict as input."""
        device = next(self.parameters()).device
        
        # Create membership vector safely using the renamed helper function.
        membership_list = get_membership_vector(membership_dict)
        membership_vector = torch.tensor(membership_list, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            self.eval() # Set model to evaluation mode
            probs, _ = self.forward(membership_vector.unsqueeze(0))
            self.train() # Set back to train mode
            selected_idx = torch.argmax(probs[0]).item()
            
        return selected_idx, membership_dict

    def train_ppo(self, memberships: torch.Tensor, actions: torch.Tensor, 
                  old_probs: torch.Tensor, rewards: torch.Tensor, 
                  clip_epsilon: float = 0.2) -> float:
        """MODIFIED: Renamed 'states' to 'memberships'."""
        memberships = torch.clamp(memberships, 0.0, 1.0)
        rewards = torch.clamp(rewards, -1.0, 1.0) # Reward is in [-1, 1]
        old_probs = torch.clamp(old_probs, 1e-6, 1.0)
        
        new_action_probs, current_values = self.forward(memberships)
        
        new_p_chosen = new_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ratio = (new_p_chosen + 1e-8) / (old_probs + 1e-8)
        
        advantages = rewards - current_values.squeeze(1).detach()
        normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        surr1 = ratio * normalized_advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * normalized_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = nn.functional.huber_loss(current_values.squeeze(1), rewards, reduction='mean', delta=1.0)
        
        total_loss = policy_loss + 0.5 * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        loss_info = f"Loss: {total_loss.item():.4f}"
        print(loss_info)
        weight_info = self.knowledge_controller.get_rule_weights_info()
        print(weight_info)

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(loss_info + '\n')
                f.write(weight_info + '\n')
        
        return total_loss.item()

    def save_model(self, file_path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'rules': self.knowledge_controller.rules,
            'membership_dim': self.membership_dim, # Renamed
            'action_dim': self.action_dim,
            'temperature': self.temperature,
            'action_names': self.action_names,
        }, file_path)
        print(f" {file_path}")
    
    @classmethod
    def load_model(cls, file_path: str, device: str = "cpu"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f": {file_path}")
        
        checkpoint = torch.load(file_path, map_location=torch.device(device))
        model = cls(
            rules=checkpoint['rules'],
            membership_dim=checkpoint['membership_dim'], # Renamed
            action_dim=checkpoint['action_dim'],
            temperature=checkpoint['temperature'],
            device=device
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.action_names = checkpoint['action_names']
        model.to(device)
        print(f" {file_path} ")
        return model

# ============== DATA LOADING AND TRAINING ADAPTER (MODIFIED) ==============
def load_data(file_path: str) -> List[Dict]:
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.lower().endswith(".jsonl"):
                print(f": {file_path}")
                for line in f:
                    if line.strip():
                        try:
                            data_list.append(json.loads(line))
                        except json.JSONDecodeError:
                            print(f" {line[:50]}...")
            else:
                print(f": {file_path}")
                try:
                    data_list = json.load(f)
                    if not isinstance(data_list, list):
                        data_list = [data_list]
                except json.JSONDecodeError as e:
                    print(f": {str(e)}")
        return data_list
    except Exception as e:
        return []

class TrainingPhaseAdapter:
    @staticmethod
    def collect_training_data(train_data: List[Dict], kogun: KoGuN, output_file: str = "training_data.json") -> None:
        """Collects training data using the membership dict directly."""
        if not train_data:
            return

        device = next(kogun.parameters()).device
        output_data = []
        for i, sample in enumerate(train_data):
            question = sample.get("question", f" {i+1}")
            # Get membership dictionary directly from input data.
            membership_dict = sample.get("membership", {})

            if not membership_dict:
                continue

            try:
                action_idx, _ = kogun.get_action(membership_dict)
                action_name = kogun.action_names[action_idx]

                # Create membership vector safely using the helper function.
                membership_list = get_membership_vector(membership_dict)
                membership_tensor = torch.tensor(membership_list, device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    kogun.eval()
                    probs, _ = kogun(membership_tensor.unsqueeze(0))
                    kogun.train()
                    selected_prob = probs[0, action_idx].item()

                output_data.append({
                    "question": question,
                    "membership": membership_dict, # Store original membership dict.
                    "action_description": action_name,
                    "old_action_prob": round(selected_prob, 6)
                })

            except Exception as e:
                print(f"'{question}' : {str(e)}")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
    @staticmethod
    def execute_actions_via_api(training_file: str, result_file: str, api_key: str) -> None:
        try:
            with open(training_file, "r", encoding="utf-8") as f:
                training_data = json.load(f)
        except Exception as e:
            return
        
        if not training_data:
            print("empty ")
            return
        
        executor = ActionExecutor(api_key)
        results = []
        mem_eval_map = {}
        if MEM_EVAL_ENABLED:
            try:
                with open(MEM_EVAL_PATH, "r", encoding="utf-8") as f:
                    mem_eval_list = json.load(f)
                if isinstance(mem_eval_list, list):
                    mem_eval_map = {
                        item.get("question"): item.get("membership", {})
                        for item in mem_eval_list
                        if isinstance(item, dict) and "question" in item
                    }
                print(f"{len(mem_eval_map)}")
            except Exception as e:
                print(f"{MEM_EVAL_PATH}: {e}")
        
        for i, entry in enumerate(training_data):
            question = entry.get("question", f" {i+1}")
            action = entry.get("action_description", "")
            
            
            try:
                if action == "1":
                    result = executor.break_into_subproblems(question)
                elif action == "2":
                    result = executor.identify_error_and_step(question)
                elif action == "3":
                    result = executor.extract_entities_and_triples(question)
                elif action == "4":
                    result = executor.provide_reasoning_chain_example(question)
                else:
                    print(f": {action}")
                    continue
                
                try:
                    cleaned_result = executor.clean_and_extract_json(result)
                    parsed_result = _first_dict(json.loads(cleaned_result))
                except json.JSONDecodeError:
                    parsed_result = {"raw_output": result}
                
                membership_result = None
                try:
                    op_condition = parsed_result.get("operation_condition", "")
                    calc_process = parsed_result.get("calculation_process", "")
                    answer_val = parsed_result.get("answer", "")
                    
                    if op_condition and calc_process and answer_val:
                        membership_eval = executor.evaluate_membership(
                            question, op_condition, calc_process, answer_val
                        )
                        try:
                            cleaned_membership = executor.clean_and_extract_json(membership_eval)
                            membership_result = _first_dict(json.loads(cleaned_membership))
                        except json.JSONDecodeError:
                            membership_result = {"error": "Invalid JSON response from membership evaluation"}
                    else:
                        membership_result = {
                            "error": "Action result is missing fields required for membership evaluation"
                        }
                except Exception as e:
                    membership_result = {"error": str(e)}

                membership_dict = _first_dict(membership_result) if membership_result else {}
                final_membership = membership_dict.get("membership", {}) if membership_dict else {}
                if MEM_EVAL_ENABLED and mem_eval_map:
                    mem_membership = mem_eval_map.get(question)
                    if mem_membership:
                        final_membership = mem_membership

                result_entry = {
                    "question": question,
                    "operation_condition": parsed_result.get("operation_condition", ""),
                    "calculation_process": parsed_result.get("calculation_process", ""),
                    "answer": parsed_result.get("answer", ""),
                    "membership": final_membership,
                }
                results.append(result_entry)
                print(f"action")
            except Exception as e:
                print(f"action error: {str(e)}")
                results.append({
                    "question": question,
                    "action": action,
                    "error": str(e),
                    "timestamp": time.time()
                })
            time.sleep(1)
        
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        

    @staticmethod
    def fill_rewards_from_result(training_file: str, result_file: str, log_file: str) -> None:
        """Fills rewards based on the 'membership' key in the result file."""
        try:
            with open(training_file, "r", encoding="utf-8") as f:
                training_data = json.load(f)
        except Exception as e:
            print(f"{training_file}: {str(e)}")
            return
    
        if not training_data:
            return
    
        results = load_data(result_file)
        if not results:
            print(f": {result_file}")
            return
        
        result_map = {r.get("question", ""): r for r in results}
        matched = 0
        not_matched = 0
        
        with open(log_file, "a", encoding="utf-8") as log_f:
            for entry in training_data:
                question = entry.get("question", "")
                original_membership = entry.get("membership", {})
                result = result_map.get(question)
        
                if result and question:
                    new_membership = result.get("membership", {})
                    if not new_membership:
                        entry["reward"] = 0.0
                        not_matched +=1
                        continue
        
                    log_f.write("\n" + "="*50 + "\n")
                    log_f.write(f"{question}\n")
                    log_f.write(json.dumps(original_membership, ensure_ascii=False, indent=4) + "\n")
                    log_f.write(jsonson.dumps(new_membership, ensure_ascii=False, indent=4) + "\n")
                    
                    # Store the new membership from the results file for the next training step.
                    entry["membership"] = new_membership
                    # compute_reward now correctly processes the result dictionary
                    entry["reward"] = compute_reward(result)
                    matched += 1
                else:
                    entry["reward"] = 0.0 # Use 0 for no match.
                    not_matched += 1
                
        with open(training_file, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)



    @staticmethod
    def update_parameters(kogun: KoGuN, training_data_file: str, log_file: str):
        """Updates parameters using membership dicts from training file."""
        
        try:
            with open(training_data_file, "r", encoding="utf-8") as f:
                training_data = json.load(f)
        except Exception as e:

            return
        
        if not training_data:
            return
        
        device = next(kogun.parameters()).device
        memberships, actions, old_probs, rewards = [], [], [], []
        
        for i, entry in enumerate(training_data):
            try:
                # Extract membership vector safely using the helper function.
                membership_dict = entry.get("membership", {})
                membership_list = get_membership_vector(membership_dict)
                
                action_name = entry["action_description"]
                action_idx = kogun.action_names.index(action_name)
                
                selected_prob = float(entry.get("old_action_prob", 0.5))
                reward_val = entry.get("reward", 0.0)
                
                memberships.append(membership_list)
                actions.append(action_idx)
                old_probs.append(selected_prob)
                rewards.append(reward_val)
                
            except Exception as e:
                continue
        
        if not memberships:
            return
            
        memberships_tensor = torch.tensor(memberships, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        old_probs_tensor = torch.tensor(old_probs, dtype=torch.float32, device=device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        
        kogun.train() # Ensure model is in training mode
        kogun.train_ppo(memberships_tensor, actions_tensor, old_probs_tensor, rewards_tensor)

def train_model():
    print("\n" + "="*50)
    print("="*50)
    
    # ============ RULE DEFINITION (UPDATED) ============
    # Value is True for 'IS' and False for 'IS NOT'.
    rules = [
        (
            {
                "Operational Accuracy": True,
                "Information Extraction Fidelity": False,
                "Intermediate Step Validity": True,
                "Answer Conformity": False
            },
        ),
        (
            {
                "Operational Accuracy": True,
                "Information Extraction Fidelity": True,
                "Intermediate Step Validity": False,
                "Answer Conformity": False
            },
        ),
        (
            {
                "Operational Accuracy": True,
                "Information Extraction Fidelity": True,
                "Intermediate Step Validity": True,
                "Answer Conformity": False
            },
        ),
        (
            {
                "Operational Accuracy": False,
                "Information Extraction Fidelity": True,
                "Intermediate Step Validity": True,
                "Answer Conformity": False
            },
        )
]
    
    # File paths
    train_file = r"" # Your initial data file
    training_data_file = ""
    result_file = "" # Your manually corrected file
    log_file = ""
    model_save_path = ""

    gemini_api_key = "" 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{device}")
    

    if os.path.exists(model_save_path):
        print(f": {model_save_path}")
        try:
            kogun_model = KoGuN.load_model(model_save_path, device=device)

            if len(kogun_model.knowledge_controller.rules) != len(rules):
                membership_dim = len(MEMBERSHIP_KEYS_ORDERED)
                action_dim = len(list(dict.fromkeys([rule[1] for rule in rules])))
                kogun_model = KoGuN(
                    rules=rules,
                    membership_dim=membership_dim,
                    action_dim=action_dim,
                    device=device,
                )
        except Exception as e:
            membership_dim = len(MEMBERSHIP_KEYS_ORDERED)
            action_dim = len(list(dict.fromkeys([rule[1] for rule in rules])))
            kogun_model = KoGuN(rules=rules, membership_dim=membership_dim, action_dim=action_dim, device=device)
    else:
        membership_dim = len(MEMBERSHIP_KEYS_ORDERED)
        action_dim = len(list(dict.fromkeys([rule[1] for rule in rules])))
        kogun_model = KoGuN(rules=rules, membership_dim=membership_dim, action_dim=action_dim, device=device)
    # =========================================================================
    # <<< MODIFIED SECTION END >>>

    kogun_model.set_log_file(log_file)
    

    if not os.path.exists(train_file):
        return
    train_samples = load_data(train_file)
    if not train_samples:

        return

    TrainingPhaseAdapter.collect_training_data(train_samples, kogun_model, training_data_file)
    

    if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY":
        return
    
    TrainingPhaseAdapter.execute_actions_via_api(training_data_file, result_file, gemini_api_key)

    if not os.path.exists(result_file):
        return
    

    TrainingPhaseAdapter.fill_rewards_from_result(training_data_file, result_file, log_file)
    

    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\n--- epoch {epoch+1}/{num_epochs} ---")
        TrainingPhaseAdapter.update_parameters(kogun_model, training_data_file, log_file)
    

    kogun_model.save_model(model_save_path)
    

if __name__ == "__main__":
    train_model()
