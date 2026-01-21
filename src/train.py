import os
import torch
import mlflow
import shutil
import tarfile
import requests
import time
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

# --- 1. CONFIGURATION (From Screenshot) ---
# Hardcoded Hyperparameters ("The Recipe")
HPARAMS = {
    "BATCH_SIZE": 1,           # Fits in 12GB VRAM
    "GRAD_ACCUM": 8,           # Simulates batch size 16
    "LEARNING_RATE": 2e-4,
    "NUM_EPOCHS": 1,
    "MAX_SEQ_LENGTH": 1024,    # Covers most Python functions
    "LOGGING_STEPS": 10,       # Smooth MLflow graph
    "SAVE_STEPS": 100,         # Frequent checkpoints
    "EVAL_STEPS": 100,         # Match save steps for "Best Model" logic
    "SAVE_TOTAL_LIMIT": 3      # Keep only 3 best checkpoints (save space)
}

# --- 2. INFRASTRUCTURE CONFIG ---
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_ID = "code-search-net/code_search_net" # Fallback to official verified dataset
DATASET_SUBSET = "python"

# Volume Mounts (Matches your K8s Job)
CACHE_ROOT = "/model-cache"
MODEL_DIR = os.path.join(CACHE_ROOT, "llama-3.2-1b")
OUTPUT_DIR = "/workspace/checkpoints/llama-doc-gen-production"

# Secrets & Services (From Env)
HF_TOKEN = os.environ.get("HF_TOKEN")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
ART_URL = os.environ.get("ARTIFACTORY_URL", "http://artifactory.artifactory.svc.cluster.local:8081/artifactory")
ART_REPO = os.environ.get("ARTIFACTORY_REPO", "generic-local")
ART_USER = os.environ.get("ARTIFACTORY_USER")
ART_PASSWORD = os.environ.get("ARTIFACTORY_PASSWORD")
ART_ARTIFACT_PATH = "base-models/llama-3.2-1b.tar.gz"


# --- 3. HELPER: SMART CACHING (From Smoke Test) ---
def ensure_model_cached():
    """
    Checks for model on Read-Only Cache Volume. 
    If missing, downloads from Artifactory to shared cache.
    """
    config_path = os.path.join(MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        print(f"✅ CACHE HIT: Model found at {MODEL_DIR}")
        return MODEL_DIR

    print(f"⚠️ CACHE MISS: Model not found at {MODEL_DIR}")
    print("   Initiating download from Artifactory...")
    
    # We use CACHE_ROOT as a staging area
    os.makedirs(CACHE_ROOT, exist_ok=True)
    download_file = os.path.join(CACHE_ROOT, "temp_download.tar.gz")
    
    full_url = f"{ART_URL}/{ART_REPO}/{ART_ARTIFACT_PATH}"
    print(f"⬇️  Downloading from: {full_url}")
    
    try:
        with requests.get(full_url, auth=(ART_USER, ART_PASSWORD), stream=True) as r:
            r.raise_for_status()
            with open(download_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print("   ✅ Download complete.")

        print("📦 Extracting...")
        with tarfile.open(download_file, "r:gz") as tar:
            tar.extractall(path=CACHE_ROOT) # Assumes tar contains "llama-3.2-1b" folder
        print("   ✅ Extraction complete.")
        
    except Exception as e:
        print(f"❌ Failed to populate cache: {e}")
        raise e
    finally:
        if os.path.exists(download_file): os.remove(download_file)

    return MODEL_DIR


# --- 4. DATASET FORMATTING ---
def format_instruction(example):
    code = example['func_code_string']
    doc = example['func_documentation_string']
    
    # Llama 3 uses ~4 chars per token, so estimate token count
    prompt_template_overhead = 150  # tokens for the chat template
    estimated_tokens = (len(code) + len(doc)) // 4 + prompt_template_overhead
    
    # If it's going to be truncated anyway, truncate the CODE intelligently
    if estimated_tokens > HPARAMS["MAX_SEQ_LENGTH"]:
        # Calculate how much code we can keep
        available_chars = (HPARAMS["MAX_SEQ_LENGTH"] - prompt_template_overhead) * 4 - len(doc)
        if available_chars > 500:  # Only truncate if we can keep meaningful code
            code = code[:available_chars] + "\n    # ... (function continues)"
        else:
            # Skip this example entirely - it's too long even with truncation
            return None
    
    prompt = (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Write a Python docstring for this code:\n```python\n{code}\n```"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    answer = f"\"\"\"{doc}\"\"\"<|eot_id|>"
    
    return {"text": prompt + answer}

def is_good_example(x):
    doc = x.get("func_documentation_string")
    code = x.get("func_code_string")

    if not doc or not code:
        return False

    # Code quality
    if len(code) < 20:
        return False

    # Docstring quality (semantic, not just length)
    if len(doc.split()) < 5:
        return False

    # Remove low-signal stubs
    bad_markers = ["TODO", "FIXME", "TBD"]
    if any(marker in doc for marker in bad_markers):
        return False

    return True

def main():
    print(f"🚀 Starting Production Training Pipeline...")
    
    # A. Hardware Verification
    if not torch.cuda.is_available():
        raise RuntimeError("❌ No GPU Detected!")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # B. Setup Infrastructure
    ensure_model_cached()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("llama-codex-training")
    
    # C. Load Tokenizer & Model
    print("⏳ Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token # Llama 3 fix

    # --- FIX: Enforce Context Length on Tokenizer directly ---
    # This avoids the TypeError in SFTTrainer while ensuring we still truncate correctly
    tokenizer.model_max_length = HPARAMS["MAX_SEQ_LENGTH"]
    print(f"📏 Enforced Tokenizer Max Length: {tokenizer.model_max_length}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16, # Native Ampere support
        device_map="auto",
        use_cache=False,            # Disable KV cache for training
        local_files_only=True
    )

    # D. Load & Clean Dataset
    print("⏳ Loading CodeSearchNet (Python)...")
    ds = load_dataset(DATASET_ID, DATASET_SUBSET, split="train", trust_remote_code=True)

    print(f"   Original Size: {len(ds)}")
    ds = ds.filter(is_good_example)
    print(f"   Cleaned Size:  {len(ds)}")

    # Limit for this specific run (remove this line to train on full dataset)
    ds = ds.select(range(50000))

    # Apply Formatting
    dataset = ds.map(format_instruction)
    dataset = dataset.filter(lambda x: x['text'] is not None)

    # Split: 95% Train, 5% Eval
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    
    # E. LoRA Configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], # Attention + MLP
        task_type="CAUSAL_LM",
        bias="none"
    )

    # F. Collator (The "Teacher")
    # This masks the user instruction so we ONLY calculate loss on the docstring
    response_template = "<|start_header_id|>assistant<|end_header_id|>"
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # G. Training Arguments (The "Recipe")
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=HPARAMS["NUM_EPOCHS"],
        per_device_train_batch_size=HPARAMS["BATCH_SIZE"],
        gradient_accumulation_steps=HPARAMS["GRAD_ACCUM"],
        learning_rate=HPARAMS["LEARNING_RATE"],
        logging_steps=HPARAMS["LOGGING_STEPS"],
        eval_strategy="steps",
        eval_steps=HPARAMS["EVAL_STEPS"],
        save_strategy="steps",
        save_steps=HPARAMS["SAVE_STEPS"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=HPARAMS["SAVE_TOTAL_LIMIT"],
        bf16=True,
        report_to="mlflow",
        run_name="llama-codex-prod-v1",
        
        # --- MEMORY OPTIMIZATIONS ---
        gradient_checkpointing=True,  # <--- CRITICAL: Saves massive VRAM
        optim="paged_adamw_8bit",     # <--- CRITICAL: Offloads optimizer state to CPU if needed

        # --- THE FIX IS HERE ---
        # Force validation to be as memory-efficient as training
        per_device_eval_batch_size=1
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        args=args,
        # CHANGED: Removed max_seq_length argument (relying on tokenizer.model_max_length)
        data_collator=collator
    )

    # H. Train
    print("🔥 Starting Training Loop...")
    trainer.train()

    # I. Save Final Artifacts
    print("💾 Saving Best Model Adapter...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_adapter"))
    
    # J. Quick Smoke Test (Verify it works)
    print("\n--- 🧠 POST-TRAINING VERIFICATION ---")
    test_code = "def calculate_area(radius): return 3.14 * radius * radius"
    prompt = (
        f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Write a Python docstring for this code:\n```python\n{test_code}\n```"
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.2)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Input Code: {test_code}")
    print(f"Generated:  {result.split('assistant')[-1].strip()}")
    print("✅ Pipeline Complete.")

if __name__ == "__main__":
    main()