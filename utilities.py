import wandb
import torch

from huggingface_hub import login, create_repo, upload_file

def upload_dataset_or_model_to_huggingface(token="your-hf-token", 
                                repo_id="dtian09/MS_MARCO_upload",
                                repo_type="dataset",
                                model_or_data_pt="skip_gram_model.pt"):
    # Step 1: Log in
    login(token=token)

    # Step 2: Create repo (skip if already created)
    repo_id = repo_id
    create_repo(repo_id, repo_type=repo_type, private=False)

    # Step 3: Upload files
    upload_file(
        path_or_fileobj=model_or_data_pt,
        path_in_repo=model_or_data_pt,
        repo_id=repo_id,
        repo_type=repo_type
    )
    print("Upload complete!")

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
