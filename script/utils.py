from pathlib import Path
import random
import json
import pandas as pd
import numpy as np
import torch


def save_jsonl(data: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


    if type(data) != pd.DataFrame:
        data = pd.DataFrame(data)
    data.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
    )

def save_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path) -> pd.DataFrame:
    path = Path(path)
    return pd.read_json(path, lines=True)
    
def log(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df: pd.DataFrame = pd.read_csv(path)
        df = pd.DataFrame(df.to_dict("records") + [data])
        df.to_csv(path, index=False)
    else:
        pd.DataFrame([data]).to_csv(path, index=False)

def save_config(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data: dict = data.to_dict()
    except:
        pass
    
    if type(data) != dict:
        data: dict = vars(data)

    for k, v in data.items():
        if type(v) != int or type(v) != str:
            data[k] = str(v)
            
    save_json(data, path)
        
def set_seed(seed: int = None) -> None:
    if seed is None:
        return 
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)