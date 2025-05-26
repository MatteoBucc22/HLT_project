# hf_utils.py
# NOTA: Upload su Hugging Face disabilitato - salvataggio solo locale
import os
# from huggingface_hub import login, create_repo, upload_folder

def hf_login():
    """Login su Hugging Face - DISABILITATO"""
    print("⚠️  Login su Hugging Face disabilitato")
    # token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
    # if not token:
    #     raise ValueError("❌ HUGGINGFACE_HUB_TOKEN not found")
    # login(token)

def save_to_hf(local_dir, repo_id="MatteoBucc/passphrase-identification", commit_msg="Upload from Colab"):
    """Upload su Hugging Face - DISABILITATO. Il modello è salvato solo localmente."""
    print(f"⚠️  Upload su Hugging Face disabilitato. Modello salvato solo localmente in: {local_dir}")
    print(f"📁 Per caricare manualmente: repo_id avrebbe dovuto essere {repo_id}")
    
    # Codice di upload commentato:
    # hf_login()
    # try:
    #     create_repo(repo_id, exist_ok=True)
    # except Exception as e:
    #     print("⚠️ Repo already exists or error:", e)

    # print("⏳ Uploading to Hugging Face Hub…")
    # upload_folder(
    #     folder_path=local_dir,
    #     repo_id=repo_id,
    #     commit_message=commit_msg,
    #     ignore_patterns=["*.pth"],
    #     repo_type="model"
    # )
    # print(f"✅ Uploaded to https://huggingface.co/{repo_id}")
