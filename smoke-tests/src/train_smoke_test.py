import os
import torch
import mlflow
import boto3
import shutil
import tarfile
import requests
import time
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# 1. Output Volume (Checkpoints - 50GB)
# We will verify we can write here
CHECKPOINT_DIR = "/workspace/checkpoints"

# 2. Input Volume (Model Cache - 20GB)
# We will verify/download the model here
CACHE_ROOT = "/model-cache"
MODEL_DIR = os.path.join(CACHE_ROOT, "llama-3.2-1b")

# Fail Fast: Secrets
HF_TOKEN = os.environ["HF_TOKEN"]
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]
S3_ENDPOINT = os.environ["MLFLOW_S3_ENDPOINT_URL"]
S3_ACCESS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
S3_SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]

# Artifactory Config
ART_URL = os.environ.get("ARTIFACTORY_URL", "http://artifactory.artifactory.svc.cluster.local:8081/artifactory")
ART_REPO = os.environ.get("ARTIFACTORY_REPO", "generic-local")
ART_USER = os.environ["ARTIFACTORY_USER"]
ART_PASSWORD = os.environ["ARTIFACTORY_PASSWORD"]
ART_ARTIFACT_PATH = "base-models/llama-3.2-1b.tar.gz"

def check_gpu_and_cuda():
    print("--- GPU and CUDA Check ---")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ FATAL: No GPU detected. Check NVIDIA device plugin in K8s.")
    
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU Detected: {device_name}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    
    if torch.cuda.is_bf16_supported():
        print("✅ Bfloat16 Support: ENABLED")
        return torch.bfloat16
    else:
        print("⚠️ Bfloat16 Support: DISABLED")
        return torch.float16

def check_checkpoint_write():
    print(f"\n--- 💾 Checkpoint Storage Write Test ---")
    print(f"   Target: {CHECKPOINT_DIR}")
    
    test_file = os.path.join(CHECKPOINT_DIR, "smoke_test_write.txt")
    try:
        # Write
        with open(test_file, "w") as f:
            f.write("NVMe Write Test Successful")
        
        # Read
        with open(test_file, "r") as f:
            content = f.read()
            
        if content == "NVMe Write Test Successful":
            print(f"✅ Write/Read Verification Successful")
        else:
            raise RuntimeError("❌ Content Mismatch on Storage Read")
            
        # Cleanup
        os.remove(test_file)
        
    except Exception as e:
        raise RuntimeError(f"❌ Storage Write Failed: {e}")

def check_services():
    print("\n--- 🔌 Service Connectivity & Logging Check ---")
    
    # 1. MLflow (Log a Metric)
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        exp_name = "smoke_test_v8"
        if not mlflow.get_experiment_by_name(exp_name):
            mlflow.create_experiment(exp_name)
        mlflow.set_experiment(exp_name)
        
        with mlflow.start_run(run_name="connectivity_check"):
            mlflow.log_param("test_phase", "initialization")
            mlflow.log_metric("connectivity_score", 1.0)
            
        print("✅ MLflow: Connected & Metric Logged")
    except Exception as e:
        raise RuntimeError(f"❌ MLflow Failed: {e}")

    # 2. MinIO (List Buckets & Dummy Upload)
    try:
        s3 = boto3.client('s3', endpoint_url=S3_ENDPOINT,
                         aws_access_key_id=S3_ACCESS_KEY,
                         aws_secret_access_key=S3_SECRET_KEY)
        
        # List
        response = s3.list_buckets()
        buckets = [b['Name'] for b in response['Buckets']]
        print(f"✅ MinIO: Connected (Buckets: {len(buckets)})")
        
        # Optional: Verify Write Permissions if a bucket exists
        if buckets:
            test_bucket = buckets[0]
            s3.put_object(Bucket=test_bucket, Key="smoke_test_ping", Body=b"ping")
            print(f"✅ MinIO: Write Access Verified on '{test_bucket}'")
            
    except Exception as e:
        raise RuntimeError(f"❌ MinIO Failed: {e}")

def check_hf_model_load():
    print("\n--- 🌐 Hugging Face Hub Auth Check ---")
    print(f"⬇️  Pinging HF Hub for {MODEL_ID}...")
    try:
        # Lighter check using Tokenizer to prove Auth works
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
        print(f"✅ HF Token Verified")
    except Exception as e:
        raise RuntimeError(f"❌ HF Auth Failed: {e}")

def ensure_model_cached():
    """
    Implements 'Lazy Loading' Cache Pattern on the READ-ONLY volume.
    """
    print(f"\n--- 📦 Smart Cache Check (Input Volume) ---")
    
    config_path = os.path.join(MODEL_DIR, "config.json")
    
    if os.path.exists(config_path):
        print(f"✅ CACHE HIT: Model found at {MODEL_DIR}")
        return MODEL_DIR

    print(f"⚠️ CACHE MISS: Model not found at {MODEL_DIR}")
    print("   Initiating download from Artifactory...")
    
    # Use CACHE_ROOT for temp files to ensure we are on the 20GB volume
    download_file = os.path.join(CACHE_ROOT, "temp_download.tar.gz")
    temp_extract_dir = os.path.join(CACHE_ROOT, "temp_extract_folder")
    
    if os.path.exists(temp_extract_dir): shutil.rmtree(temp_extract_dir)
    os.makedirs(temp_extract_dir, exist_ok=True)
    
    # A. Download
    full_url = f"{ART_URL}/{ART_REPO}/{ART_ARTIFACT_PATH}"
    print(f"⬇️  Downloading from: {full_url}")
    
    try:
        with requests.get(full_url, auth=(ART_USER, ART_PASSWORD), stream=True) as r:
            r.raise_for_status()
            with open(download_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)
        print("   ✅ Download complete.")

        # B. Extract
        print("📦 Extracting...")
        with tarfile.open(download_file, "r:gz") as tar:
            tar.extractall(path=temp_extract_dir)
        print("   ✅ Extraction complete.")

        # C. Atomic Move
        if os.path.exists(MODEL_DIR): shutil.rmtree(MODEL_DIR)
        shutil.move(temp_extract_dir, MODEL_DIR)
        print(f"   ✅ Installed model to {MODEL_DIR}")

    except Exception as e:
        print(f"❌ Failed to populate cache: {e}")
        if os.path.exists(temp_extract_dir): shutil.rmtree(temp_extract_dir)
        raise e
    finally:
        if os.path.exists(download_file): os.remove(download_file)

    return MODEL_DIR

def run_inference_test(model_path, dtype):
    print(f"\n--- 🧠 Verification & Inference Test ---")
    try:
        print("   Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        # Fix for Llama missing pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("   Loading Model (Offline)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=dtype, 
            local_files_only=True,
            device_map="auto"
        )
        print(f"   ✅ Model Loaded on {model.device}")

        # INFERENCE CALCULATION
        print("   🧪 Running Test Inference...")
        input_text = "The capital of France is"
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5)
            
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   📝 Input: '{input_text}'")
        print(f"   📝 Output: '{result}'")
        print("   ✅ CUDA/Matrix-Multiplication Verified.")
        
    except Exception as e:
        raise RuntimeError(f"❌ Inference Test Failed: {e}")

def main():
    print("🚀 Starting Supply Chain & Hardware Verification (v8)...")
    
    # 1. Hardware Check
    dtype = check_gpu_and_cuda()
    
    # 2. Storage Write Check (Checkpoints PVC)
    check_checkpoint_write()
    
    # 3. Connectivity & Logging (MLflow/MinIO)
    check_services()
    
    # 4. Auth Check (Hugging Face)
    check_hf_model_load()
    
    # 5. Cache Population (Model Cache PVC)
    model_path = ensure_model_cached()
    
    # 6. Final Logic Verification
    run_inference_test(model_path, dtype)
    
    print("\n✅✅✅ SMOKE TEST PASSED: Infrastructure is ready for training.")

if __name__ == "__main__":
    main()