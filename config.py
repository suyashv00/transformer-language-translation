from pathlib import Path

def get_config():
    return {
        "batch_size": 8, 
        "num_epochs": 30,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "english",
        "lang_tgt": "marathi",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "checkpoint_every_steps": 300
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = Path(config['model_folder'])
    weights_files = list(model_folder.glob(f"{config['model_basename']}*.pt"))
    if not weights_files:
        return None
    return str(max(weights_files, key=lambda p: p.stat().st_mtime))

