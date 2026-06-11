
import os
import re
import copy
import math
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from datetime import datetime

from collections import deque

from reward_process import location_reward , type_reward , description_reward

logger = logging.getLogger(__name__)

# ── Reasoning Judge (HTTP client to judge_server.py) ────────────────────────
# Supports multiple judge URLs for parallel scoring (comma-separated)
# e.g. JUDGE_URLS="http://127.0.0.1:5100,http://127.0.0.1:5101"

JUDGE_URLS = [u.strip() for u in os.environ.get("JUDGE_URLS", os.environ.get("JUDGE_URL", "http://127.0.0.1:5100")).split(",")]

# Gemini vision judge URL (for grouped scoring with images)
GEMINI_JUDGE_URL = os.environ.get("GEMINI_JUDGE_URL", "http://127.0.0.1:5200")

# Number of generations per prompt (set from training args, default 4)
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "4"))


def _call_judge_single_server(url, items):
    """Send a batch to a single judge server. Returns list of scores."""
    endpoint = f"{url}/batch"
    data = json.dumps({"items": items}).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            return result.get("scores", [0.5] * len(items))
    except Exception as e:
        logger.warning(f"[JUDGE] HTTP call to {url} failed: {e}")
        return [0.5] * len(items)


def _call_judge_batch(items):
    """Split items across available judge servers and score in parallel."""
    if not items:
        return []
    n_servers = len(JUDGE_URLS)
    if n_servers == 1:
        return _call_judge_single_server(JUDGE_URLS[0], items)

    # Split items round-robin across servers
    chunks = [[] for _ in range(n_servers)]
    index_map = [[] for _ in range(n_servers)]  # track original indices
    for i, item in enumerate(items):
        server_idx = i % n_servers
        chunks[server_idx].append(item)
        index_map[server_idx].append(i)

    # Call all servers in parallel
    with ThreadPoolExecutor(max_workers=n_servers) as executor:
        futures = []
        for server_idx in range(n_servers):
            if chunks[server_idx]:
                futures.append((server_idx, executor.submit(_call_judge_single_server, JUDGE_URLS[server_idx], chunks[server_idx])))

    # Reassemble scores in original order
    scores = [0.5] * len(items)
    for server_idx, future in futures:
        chunk_scores = future.result()
        for local_i, score in enumerate(chunk_scores):
            scores[index_map[server_idx][local_i]] = score

    return scores


def reasoning_reward(completions, solution, **kwargs):
    """Judge the quality of <think> reasoning via external judge server."""
    contents = [completion[0]["content"] for completion in completions]
    product_list = kwargs.get("product", [])
    items = []
    valid_indices = []

    for i, (content, sol) in enumerate(zip(contents, solution)):
        # Extract think block
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
        if not think_match or len(think_match.group(1).strip()) < 20:
            continue

        # Extract GT info from solution
        sol_answer_match = re.search(r'<answer>(.*?)</answer>', sol)
        gt_label = sol_answer_match.group(1).strip().lower() if sol_answer_match else "no"
        gt_type_match = re.search(r'<type>(.*?)</type>', sol)
        gt_type = gt_type_match.group(1).strip() if gt_type_match else ""
        gt_loc_match = re.search(r'<location>(.*?)</location>', sol)
        gt_location = gt_loc_match.group(1).strip() if gt_loc_match else ""

        # Extract model answer
        model_ans_match = re.search(r'<answer>(.*?)</answer>', content)
        model_answer = model_ans_match.group(1).strip() if model_ans_match else ""

        product = product_list[i] if i < len(product_list) else "unknown"

        items.append({
            "think_text": think_match.group(1).strip(),
            "gt_label": gt_label,
            "gt_type": gt_type,
            "gt_location": gt_location,
            "product": product,
            "model_answer": model_answer,
        })
        valid_indices.append(i)

    # Batch call to judge
    scores = _call_judge_batch(items) if items else []

    # Build rewards array: 0.0 for invalid, scored for valid
    rewards = [0.0] * len(contents)
    for idx, score in zip(valid_indices, scores):
        rewards[idx] = score

    return rewards

GEMINI_CLIENT_MAX_RETRIES = 3
GEMINI_CLIENT_TIMEOUT = 180  # server-side retry cascade can take ~120s

def _call_gemini_judge_grouped(groups):
    """Send grouped completions to Gemini vision judge server with retries."""
    endpoint = f"{GEMINI_JUDGE_URL}/batch_grouped"
    data = json.dumps({"groups": groups}).encode("utf-8")

    for attempt in range(GEMINI_CLIENT_MAX_RETRIES):
        try:
            req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=GEMINI_CLIENT_TIMEOUT) as resp:
                raw = resp.read()
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    logger.error(f"[GEMINI-JUDGE] JSON decode failed, raw response: {raw[:500]}")
                    continue
                scores = result.get("group_scores", [])
                if not scores:
                    logger.warning(f"[GEMINI-JUDGE] Empty group_scores, raw response: {raw[:500]}")
                return scores
        except Exception as e:
            logger.warning(f"[GEMINI-JUDGE] HTTP call failed (attempt {attempt+1}/{GEMINI_CLIENT_MAX_RETRIES}): {e}")
            if attempt < GEMINI_CLIENT_MAX_RETRIES - 1:
                import time as _time
                _time.sleep(1)
    logger.error(f"[GEMINI-JUDGE] All {GEMINI_CLIENT_MAX_RETRIES} HTTP retries exhausted")
    return None


def reasoning_reward_gemini(completions, solution, **kwargs):
    """Judge reasoning quality via Gemini with vision — grouped per prompt."""
    contents = [completion[0]["content"] for completion in completions]
    product_list = kwargs.get("product", [])
    image_path_list = kwargs.get("image_path", [])

    n = len(contents)
    num_gen = NUM_GENERATIONS

    # Group completions by prompt (every num_gen consecutive entries share a prompt)
    num_prompts = n // num_gen
    groups = []
    group_valid_indices = []  # track which completions in each group are valid

    for g in range(num_prompts):
        start = g * num_gen
        end = start + num_gen

        # Extract GT info from the first solution in the group (all same)
        sol = solution[start]
        sol_answer_match = re.search(r'<answer>(.*?)</answer>', sol)
        gt_label = sol_answer_match.group(1).strip().lower() if sol_answer_match else "no"
        gt_type_match = re.search(r'<type>(.*?)</type>', sol)
        gt_type = gt_type_match.group(1).strip() if gt_type_match else ""
        gt_loc_match = re.search(r'<location>(.*?)</location>', sol)
        gt_location = gt_loc_match.group(1).strip() if gt_loc_match else ""

        product = product_list[start] if start < len(product_list) else "unknown"
        image_path = image_path_list[start] if start < len(image_path_list) else ""

        # Build completions for this group
        group_completions = []
        valid_mask = []
        for i in range(start, end):
            content = contents[i]
            think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
            model_ans_match = re.search(r'<answer>(.*?)</answer>', content)
            model_answer = model_ans_match.group(1).strip() if model_ans_match else ""

            if think_match and len(think_match.group(1).strip()) >= 20:
                group_completions.append({
                    "think_text": think_match.group(1).strip(),
                    "model_answer": model_answer,
                })
                valid_mask.append(True)
            else:
                # Still include a placeholder so indices align
                group_completions.append({
                    "think_text": "(no reasoning provided)",
                    "model_answer": model_answer,
                })
                valid_mask.append(False)

        # Derive GT mask path for RealIAD: same filename, .jpg -> .png
        gt_mask_path = ""
        if gt_label == "yes" and image_path and image_path.lower().endswith(".jpg"):
            candidate = image_path[:-4] + ".png"
            if os.path.isfile(candidate):
                gt_mask_path = candidate

        groups.append({
            "image_path": image_path,
            "gt_mask_path": gt_mask_path,
            "gt_label": gt_label,
            "gt_type": gt_type,
            "gt_location": gt_location,
            "product": product,
            "completions": group_completions,
        })
        group_valid_indices.append(valid_mask)

    # Call Gemini judge with all groups
    group_scores = _call_gemini_judge_grouped(groups)

    # Build flat rewards array
    rewards = [0.0] * n
    if group_scores and len(group_scores) == num_prompts:
        for g in range(num_prompts):
            start = g * num_gen
            scores = group_scores[g]
            valid_mask = group_valid_indices[g]
            for j in range(num_gen):
                if j < len(scores) and valid_mask[j]:
                    rewards[start + j] = scores[j]
                # invalid completions stay at 0.0
    else:
        # Fallback: 0.5 for all valid completions
        logger.warning(f"[GEMINI-JUDGE] Fallback — unexpected result: group_scores={group_scores}, expected {num_prompts} groups")
        for g in range(num_prompts):
            start = g * num_gen
            for j in range(num_gen):
                idx = start + j
                content = contents[idx]
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
                if think_match and len(think_match.group(1).strip()) >= 20:
                    rewards[idx] = 0.5

    return rewards

def consistency_reward(completions, solution, **kwargs):
    pattern_no = r"^(?!.*<location>)(?!.*<type>).*<think>.*?</think>\s*<answer>.*?</answer>.*$"  # normal pattern    
    pattern_yes = r".*<think>.*?</think>\s*<location>.*?</location>\s*<type>.*?</type>\s*<answer>.*?</answer>.*"  # abnormal pattern
    completion_contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for content, sol in zip(completion_contents, solution):
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()  # yes or no
        if ground_truth.lower() == "yes":
            # yes --> pattern_yes
            match_result = re.fullmatch(pattern_yes, content, re.DOTALL)
            rewards.append(1.0 if match_result else 0.0)
        elif ground_truth.lower() == "no":
            # no --> pattern_no
            match_result = re.fullmatch(pattern_no, content, re.DOTALL)
            rewards.append(1.0 if match_result else 0.0)
        else:
            rewards.append(0.0)
    return rewards

def accuracy_reward(completions, solution, **kwargs):
    """answer, location, type"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:

            #resolve ground_truth
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
            gt = ground_truth.lower()

            # gt == "no"
            if gt == "no":
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                if content_match:
                    ans = content_match.group(1).strip().lower()
                    if ans == "no":
                        reward = 1.0
            
            # gt == "yes"
            elif gt == "yes":
                total_reward = 0.0
                max_reward = 2.0 
                
                # ----- type reward -----
                gpt_type_match = re.search(r'<type>(.*?)</type>', content)
                gt_type_match = re.search(r'<type>(.*?)</type>', sol)
                if gpt_type_match and gt_type_match:
                    gpt_type = gpt_type_match.group(1).strip().lower()
                    gt_type = gt_type_match.group(1).strip().lower()
                    type_calculator = type_reward.AnomalyRewardCalculator()
                    type_score = type_calculator.compute_reward(gpt_type, gt_type)
                    total_reward += type_score
                else:
                    pass # reward += 0

                # location reward
                gpt_location_match = re.search(r'<location>(.*?)</location>', content)
                gt_location_match = re.search(r'<location>(.*?)</location>', sol)
                if gpt_location_match and gt_location_match:
                    gpt_location = gpt_location_match.group(1).strip().lower()
                    gt_location = gt_location_match.group(1).strip().lower()
                    location_score = location_reward.map_location_to_region(gpt_location , gt_location)
                    total_reward += location_score
                else:
                    pass

                reward = total_reward / max_reward # 0~1

                # answer reward
                answer_match = re.search(r'<answer>(.*?)</answer>', content)
                if answer_match:
                    ans = answer_match.group(1).strip().lower()
                    if ans == "yes":
                        reward += 1.0
                    
                
                
        except Exception:
            pass  # reward = 0   
        rewards.append(reward)            
    return rewards





def consistency_reward_cot(completions, solution, **kwargs):
    completion_contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for content, sol in zip(completion_contents, solution):

        sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.IGNORECASE)
        ground_truth = sol_match.group(1).strip().lower() if sol_match else sol.strip().lower()
        

        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.IGNORECASE)
        if not answer_match:

            rewards.append(0.0)
            continue
        
        model_answer = answer_match.group(1).strip().lower()
        

        if model_answer != ground_truth:

            rewards.append(0.0)
            continue
        

        has_type = bool(re.search(r'<type>.*?</type>', content, re.IGNORECASE | re.DOTALL))
        has_location = bool(re.search(r'<location>.*?</location>', content, re.IGNORECASE | re.DOTALL))
        has_description = bool(re.search(r'<description>.*?</description>', content, re.IGNORECASE | re.DOTALL))
        

        tag_count = sum([has_type, has_location, has_description])
        

        if model_answer == "no":

            consistency_reward = 1.0 if tag_count == 0 else 0.0
                
        elif model_answer == "yes":

            if tag_count == 3:
                consistency_reward = 1.0  
            elif tag_count == 2:
                consistency_reward = 0.7 
            elif tag_count == 1:
                consistency_reward = 0.4 
            else:
                consistency_reward = 0.0 
        else:
            consistency_reward = 0.0
        
        rewards.append(consistency_reward)
    
    return rewards

def format_consistency_reward_cot(completions, solution, **kwargs):
    
    completion_contents = [completion[0]["content"] for completion in completions]
    
    rewards = []
    for content, sol in zip(completion_contents, solution):

        sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.IGNORECASE)
        ground_truth = sol_match.group(1).strip().lower() if sol_match else sol.strip().lower()
        

        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.IGNORECASE)
        if not answer_match:

            rewards.append(0.0)
            continue
        
        model_answer = answer_match.group(1).strip().lower()
        
        if model_answer != ground_truth:
            rewards.append(0.0)
            continue
        
        has_type = bool(re.search(r'<type>.*?</type>', content, re.IGNORECASE | re.DOTALL))
        has_location = bool(re.search(r'<location>.*?</location>', content, re.IGNORECASE | re.DOTALL))
        has_description = bool(re.search(r'<description>.*?</description>', content, re.IGNORECASE | re.DOTALL))
        

        tag_count = sum([has_type, has_location, has_description])
        

        if model_answer == "no":

            format_consistency_reward = 1.0 if tag_count == 0 else 0.0
                
        elif model_answer == "yes":

            if tag_count == 3:
                format_consistency_reward = 1.0
            elif tag_count == 2:
                format_consistency_reward = 0.7
            elif tag_count == 1:
                format_consistency_reward = 0.4
            else:  # tag_count == 0
                format_consistency_reward = 0.0
        else:

            format_consistency_reward = 0.0
        
        rewards.append(format_consistency_reward)
    
    return rewards


def accuracy_reward_cot_wo_type(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
  
            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            if ground_truth.lower() == "no":

                content_match = re.search(r'<answer>(.*?)</answer>', content)

                if content_match and content_match.group(1).strip().lower() == "no":
                    reward = 1.0

            elif ground_truth.lower() == "yes":

                total_reward = 0.0
                max_reward = 1.0 


                gpt_location_match = re.search(r'<location>(.*?)</location>', content)
                gt_location_match = re.search(r'<location>(.*?)</location>', sol)
                gpt_location = gpt_location_match.group(1).strip().lower()
                gt_location = gt_location_match.group(1).strip().lower()
                location_score = location_reward.map_location_to_region(gpt_location , gt_location)
                total_reward += location_score

                reward = total_reward / max_reward


                
                answer_match = re.search(r'<answer>(.*?)</answer>', content)
                if answer_match and answer_match.group(1).strip().lower() == "yes":
                    reward += 1.0
                    
                  
                
        except Exception:
            pass 
        rewards.append(reward)            
    return rewards

def accuracy_reward_cot_wo_location(completions, solution, **kwargs):

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:

            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            if ground_truth.lower() == "no":

                content_match = re.search(r'<answer>(.*?)</answer>', content)
                if content_match and content_match.group(1).strip().lower() == "no":
                    reward = 1.0
            elif ground_truth.lower() == "yes":

                total_reward = 0.0
                max_reward = 1.0 
                
                gpt_type_match = re.search(r'<type>(.*?)</type>', content)
                gt_type_match = re.search(r'<type>(.*?)</type>', sol)
                gpt_type = gpt_type_match.group(1).strip().lower()
                gt_type = gt_type_match.group(1).strip().lower()
                type_calculator = type_reward.AnomalyRewardCalculator()
                type_score = type_calculator.compute_reward(gpt_type, gt_type)
                total_reward += type_score

                reward = total_reward / max_reward
                
                answer_match = re.search(r'<answer>(.*?)</answer>', content)
                if answer_match and answer_match.group(1).strip().lower() == "yes":
                    reward += 1.0
                
        except Exception:
            pass    
        rewards.append(reward)            
    return rewards

def format_reward_cot_base(completions, solution, **kwargs):
    pattern = r".*<think>.*?</think><answer>.*?</answer>.*" 
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(completion_contents, solution):
        sol_match = re.search(r'<answer>(.*?)</answer>', sol)
        ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
        match_result = re.fullmatch(pattern, content, re.DOTALL)
        rewards.append(1.0 if match_result else 0.0)
    return rewards

def accuracy_reward_cot_base(completions, solution, **kwargs):

    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:

            sol_match = re.search(r'<answer>(.*?)</answer>', sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            if ground_truth.lower() == "no":

                content_match = re.search(r'<answer>(.*?)</answer>', content)

                if content_match and content_match.group(1).strip().lower() == "no":
                    reward = 1.0

            elif ground_truth.lower() == "yes":

                answer_match = re.search(r'<answer>(.*?)</answer>', content)

                if answer_match and answer_match.group(1).strip().lower() == "yes":
                    reward += 1.0

        except Exception:
            pass  
        rewards.append(reward)            
    return rewards

def wo_format(completions, solution, **kwargs):
    rewards = 0
    return rewards
