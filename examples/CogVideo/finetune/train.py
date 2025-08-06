import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))

from finetune.models.utils import get_model_cls
from finetune.schemas import Args
import os
import json

def save_args_to_json(args):
    args_dict = vars(args)
    
    def path_to_str(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: path_to_str(v) for k, v in value.items()}
        if isinstance(value, list):
            return [path_to_str(v) for v in value]
        return value
    
    processed_args = path_to_str(args_dict)
    
    output_dir = processed_args["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "config.json")
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(processed_args, f, indent=2)
    
def main():
    args = Args.parse_args()
    save_args_to_json(args)
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.fit()


if __name__ == "__main__":
    main()
