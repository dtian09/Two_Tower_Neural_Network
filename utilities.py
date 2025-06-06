from transformers.tokenization_utils_base import import_protobuf_decode_error
import wandb
import torch
from huggingface_hub import HfApi, HfFolder, upload_file, login
from huggingface_hub.utils import RepositoryNotFoundError

def upload_dataset_or_model_to_huggingface(
    token="your-hf-token",
    repo_id="dtian09/MS_MARCO_upload",
    repo_type="dataset",
    model_or_data_pt="skip_gram_model.pt"
):
    # Step 1: Log in
    login(token=token)

    # Step 2: Check if repo exists, create if not
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repository '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Repository '{repo_id}' not found. Creating it now...")
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print("Repository created successfully.")

    # Step 3: Upload file
    upload_file(
        path_or_fileobj=model_or_data_pt,
        path_in_repo=model_or_data_pt,
        repo_id=repo_id,
        repo_type=repo_type
    )
    print("Upload complete!")

def download_from_huggingface(repo_id = "dtian09/MS_MARCO",
                              model_or_data_pt = "best_two_tower_model.pt"):

    from huggingface_hub import hf_hub_download

    model_or_data_path = hf_hub_download(repo_id=repo_id, filename=model_or_data_pt,  repo_type="dataset" )

    print(F"download from Hugging Face complete! {model_or_data_path}")
    return model_or_data_path   

def save_model_to_wandb( model, artifact_name: str, model_pt: str = "full_model.pt", project="your_project_name" ):
    """
    Saves a model to a W&B artifact.

    Args:
        model: The model to save.
        artifact_name (str): The name of the W&B artifact in the format "entity/project:version".
        model_pt (str): The name of the file to save the model as.

    Returns:
        None
    """ 
    # 1. Login to W&B
    wandb.login()

    # 2. Create a new W&B run
    run = wandb.init(project, job_type="save_model")

    # 3. Create an artifact
    artifact = wandb.Artifact(artifact_name, type='model')

    # 4. Save the model locally
    torch.save(model, model_pt)

    # 5. Add the model file to the artifact
    artifact.add_file(model_pt)

    # 6. Log the artifact
    run.log_artifact(artifact)

    # 7. Finish the run
    run.finish()

def reload_model_from_wandb( artifact: str, model_pt: str = "full_model.pt" ):
    """
    Reloads a model from a W&B artifact.

    Args:
        artifact (str): The W&B artifact string in the format "entity/project:version".

    Returns:
        model: The loaded model.
    """ 
  
    # 1. Login to W&B
    wandb.login()

    # 2. Load artifact
    artifact = wandb.use_artifact(artifact, type='model')

    # 3. Download it locally
    artifact_dir = artifact.download()

    # 4. Load the model
    model = torch.load(f"{artifact_dir}/{model_pt}", map_location=torch.device('cpu'))
    model.eval()
    return model

def count_rows_of_chromadb():
    import chromadb

    # Connect to the persistent ChromaDB client
    client = chromadb.PersistentClient(path="./chroma_db")

    # Load your collection
    collection = client.get_collection(name="ms_marco_passages_lora")

    # Get the number of entries
    num_rows = collection.count()

    print(f"Number of rows in the collection: {num_rows}")

def check_ANN_search_metric():
    from chromadb import PersistentClient

    # Connect to ChromaDB
    client = PersistentClient(path="./chroma_db")

    # Load your collection
    collection = client.get_collection(name="ms_marco_passages_lora")

    # Check metadata for ANN search metric
    if hasattr(collection, "metadata"):
        space = collection.metadata.get("hnsw:space", "not specified")
        print(f"ANN search metric: {space}")
    else:
        print("No metadata found. Default metric (likely L2) may be used.")

if __name__ == "__main__":
  
  #check_ANN_search_metric()

  #count_rows_of_chromadb()
  
  #before running this script, run command: export HF_TOKEN=hf_token
  '''
  import os
  from huggingface_hub import login
  
  HF_TOKEN = os.getenv("HF_TOKEN") 
  #repo_id = "dtian09/MS_MARCO"
  #repo_type="dataset"
  #model_or_data_pt = 'best_two_tower_lora.pt'
  
  repo_id = "dtian09/clip_llama"
  repo_type="model"
  model_or_data_pt = 'trained_clip_llama_50percent.zip'
  
  upload_dataset_or_model_to_huggingface(token=HF_TOKEN,
                                repo_id=repo_id,
                                repo_type=repo_type,
                                model_or_data_pt=model_or_data_pt)
 ''' 

  model_or_data_path = download_from_huggingface(repo_id = "dtian09/MS_MARCO",
                                                model_or_data_pt = "best_two_tower_lora.pt")

  print(torch.load(model_or_data_path, map_location="cpu"))

