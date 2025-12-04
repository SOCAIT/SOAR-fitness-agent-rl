import os
from huggingface_hub import HfApi, login

# Configuration
MODEL_PATH = "/Users/socait/Desktop/AGI/fitness-reasoning-rl-agent/.models/fitness-agent-langgraph-14B-qwen2.5-005/checkpoints/0037"
REPO_ID = "socaitcy/fitness-agent-14B-qwen2.5-adapter"  # Change this to your HF username and desired model name

def upload_model():
    print(f"Preparing to upload model from: {MODEL_PATH}")
    print(f"Target repository: {REPO_ID}")
    
    # Ensure you are logged in
    try:
        # This is a safer way to login programmatically or check if logged in
        # If you aren't logged in, this will prompt you in the terminal
        api = HfApi()
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception:
        print("⚠️ You are not logged in.")
        token = input("Please paste your Hugging Face Access Token (Write permission): ").strip()
        login(token=token)
        api = HfApi()

    # Create the repo if it doesn't exist
    try:
        api.create_repo(repo_id=REPO_ID, repo_type="model", private=True)
        print(f"Created repository {REPO_ID}")
    except Exception as e:
        if "401" in str(e) or "403" in str(e):
            print(f"\n❌ Permission Error: Your token does not have 'Write' access.")
            print(f"Logged in as: {api.whoami()['name']}")
            print("Please paste a new token with WRITE permissions from https://huggingface.co/settings/tokens")
            
            token = input("New Access Token: ").strip()
            if token:
                login(token=token)
                # Retry creation
                try:
                    api.create_repo(repo_id=REPO_ID, repo_type="model", private=True)
                    print(f"Created repository {REPO_ID}")
                except Exception as retry_e:
                    print(f"Retry failed: {retry_e}")
                    return
            else:
                print("No token provided. Exiting.")
                return
        elif "You already have a repository" in str(e) or "409" in str(e):
            print(f"Repository {REPO_ID} already exists. Proceeding with upload...")
        else:
            print(f"⚠️ Warning: Issue creating repository: {e}")
            print("Attempting to upload anyway (repo might already exist)...")

    # Upload the folder
    print("Uploading files...")
    api.upload_folder(
        folder_path=MODEL_PATH,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload fine-tuned LoRA adapter"
    )
    print(f"Successfully uploaded to https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
        upload_model()
    except ImportError:
        print("Please install huggingface_hub first: pip install huggingface_hub")

