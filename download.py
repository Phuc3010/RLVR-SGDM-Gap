from huggingface_hub import snapshot_download
from datasets import load_dataset, load_from_disk
import gc
from tqdm import tqdm

if __name__=="__main__":
    
    model_sizes = ["1B"]
    pt_tokens = [20, 40, 80, 160, 320]
    steps = [50, 100, 150, 200, 250, 300]

    pt_tokens = [160]
    model_size = "1B"
    # for step in steps:
        # for ds_name in ["evolm", "omega", "polaris"]:
    local_path = snapshot_download(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        # repo_id="DatPySci/RLVR-CoTs",
        # allow_patterns=[f"qwen2.5-3b/Qwen2.5-3B-{ds_name}-GRPO/global_step_{step}/actor_hf/**"],
        local_dir=f"models/Llama-3.2-3B-Instruct",
    )
    # datasets = ["omega", "evolm"]
    # local_path = snapshot_download(
    #     repo_type="model",            # or "dataset" / "space"
    #     local_dir=f"data/synthetic/SFT",
    # )

    # val_dataset = load_dataset("openai/gsm8k", "main", split=f"test")

    # train_dataset.save_to_disk("data/gsm8k_train")
    # val_dataset.save_to_disk("data/gsm8k_test")

    
    # dataset = load_dataset("meta-math/MetaMathQA", split="train")
    # dataset.save_to_disk("data/MetaMathQA")

    # for pt_tokens in tqdm(pt_tokens):
    #     for model_size in model_sizes:
    # local_path = snapshot_download(
        # repo_id=f"zhenting/evolm-{model_size}-{pt_tokens}BT-cpt-MixedFW8FM42-sftep1-sampled500k_first100k_qwen7b",
        # repo_id="meta-math/MetaMathQA",
        # repo_id="DatPySci/PreRLVR-Controlled",
        # allow_patterns=[f"models/GRPO/EvoLM-1B-160BT-{dataset}-step300/**"],
        # local_dir="models/Qwen2.5-3B",            # or "dataset" / "space"
        # repo_type="model",
        # local_dir="models/Qwen1.5-0.5B"
        # local_dir="models/Qwen2.5-3B"
        # local_dir=f"models/SFT/EvoLM-{model_size}-{pt_tokens}BT-MixedFW8FM42-100k"
    # )
            # model = AutoModelForCausalLM.from_pretrained(f"zhenting/evolm-{model_size}-{pt_tokens}BT", device_map="cpu", cache_dir=f"models/evolm-{model_size}-{pt_tokens}BT")
            # del model
            # gc.collect()
            
            # except:
            #     continue
    
    # print(f"Downloaded to: {local_path}")