# pip install transformers accelerate torch --upgrade
from typing import Dict, List, Optional
import os, sys
import json, re
from huggingface_hub import login, whoami
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline


ALLOWED_TASKS = ["push", "pull", "pick", "stack"]
TASK_SYNONYMS = {
    "push": ["push", "pushing"],
    "pull": ["pull", "pulling"],
    "pick": ["pick", "pickup", "pick up", "pick-and-place", "pick & place", "pick/place"],
    "stack": ["stack", "stacking"]
}

def normalize_task(token: str) -> Optional[str]:
    t = token.lower().strip()
    for canonical, syns in TASK_SYNONYMS.items():
        if any(t == s for s in syns):
            return canonical
    # loose match
    for canonical, syns in TASK_SYNONYMS.items():
        if any(s in t for s in syns):
            return canonical
    return None

def heuristic_parse(order_text: str, allowed: List[str] = ALLOWED_TASKS) -> Dict[str, int]:
    """
    Rule-based fallback for sequences like:
    'push first, then pull and then pick and stack'
    """
    text = order_text.lower()
    # split by common sequencing cues
    tokens = re.split(r"[,\.;]|and then|then|and|->|→|>", text)
    seq = []
    for tok in tokens:
        # grab any known task words inside this fragment
        for word in re.findall(r"[a-zA-Z\+\&\- ]+", tok):
            cand = normalize_task(word)
            if cand and cand not in seq and cand in allowed:
                seq.append(cand)
    # final pass: keep original order, assign 1..n
    return {task: i + 1 for i, task in enumerate(seq)}

def build_llm_prompt(nl_instruction: str, allowed: List[str]) -> str:
    system = (
        "You are a planner that converts natural-language instructions about task order into JSON.\n"
        "Return a single, valid JSON object (no prose) mapping tasks to integer priorities starting at 1.\n"
        "Lower numbers mean earlier steps. Do not include explanations.\n"
        f"Allowed tasks only: {allowed}.\n"
        'Example input: "push first, then pull, then pick and stack"\n'
        'Example output: {"push": 1, "pull": 2, "pick": 3, "stack": 4}'
    )
    user = f"Instruction: {nl_instruction}\nOutput JSON only."
    # Many Llama-Instruct chat templates expect system + user messages.
    return f"<<SYS>>\n{system}\n<</SYS>>\n\n[INST] {user} [/INST]"

class TaskPlannerLLM:
    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct", token = "", device_map: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code = True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, token=token , dtype="auto", device_map=device_map, trust_remote_code = True)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False
        )

    def plan(self, nl_instruction: str, allowed: List[str] = ALLOWED_TASKS) -> Dict[str, int]:
        prompt = build_llm_prompt(nl_instruction, allowed)
        out = self.pipe(
            prompt,
            max_new_tokens=128,
            do_sample=False,   # deterministic
            temperature=0.0
        )[0]["generated_text"].strip()

        # Try strict JSON first
        try:
            parsed = json.loads(out)
            items = [(k, v) for k, v in parsed.items() if normalize_task(k) in allowed]
            items.sort(key=lambda kv: (int(kv[1]),))
            # rebuild with canonical keys and 1..n indexing
            ordered_unique = []
            seen = set()
            for k, _ in items:
                canon = normalize_task(k)
                if canon and canon not in seen:
                    ordered_unique.append(canon)
                    seen.add(canon)
            return {task: i + 1 for i, task in enumerate(ordered_unique)} or heuristic_parse(nl_instruction, allowed)
        except Exception:
            return heuristic_parse(nl_instruction, allowed)

# --- Example usage ---
if __name__ == "__main__":
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    HF_TOKEN = os.getenv("HF_TOKEN")

    planner = TaskPlannerLLM(model_id=MODEL_ID, token=HF_TOKEN)
    instruction = "push first, and then pull and then pick and stack"
    plan = planner.plan(instruction)
    print(plan)  # e.g., {'push': 1, 'pull': 2, 'pick': 3, 'stack': 4}

    # Optional: map to environments/actions
    task_to_env = {
        "push": "PushCube-v1",
        "pull": "PullCube-v1",
        "pick": "PickCube-v1",
        "stack": "StackCube-v1",
    }
    ordered_envs = [task_to_env[t] for t, _ in sorted(plan.items(), key=lambda kv: kv[1])]
    print("Sequence:", ordered_envs)
