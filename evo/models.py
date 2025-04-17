import pkgutil
import re
import os
from transformers import AutoConfig, AutoModelForCausalLM
import yaml
import torch

from stripedhyena.utils import dotdict
from stripedhyena.model import StripedHyena
from stripedhyena.tokenizer import CharLevelTokenizer


MODEL_NAMES = [
    'evo-1.5-8k-base',
    'evo-1-8k-base',
    'evo-1-131k-base',
    'evo-1-8k-crispr',
    'evo-1-8k-transposon',
]

class Evo:
    def __init__(self, model_name: str = MODEL_NAMES[1], device: str = None, local_files_only: bool = False):
        """
        Loads an Evo model checkpoint given a model name.
        If the checkpoint does not exist, we automatically download it from HuggingFace.
        
        Args:
            model_name: Name of the Evo model to load or path to a locally downloaded model
            device: Device to load the model on
            local_files_only: If True, won't attempt to download the model and will use only local files
        """
        self.device = device

        # Check if model_name is a path
        if os.path.exists(model_name) and os.path.isdir(model_name):
            is_local_path = True
            print(f"Using local model path: {model_name}")
            config_path = 'configs/evo-1-8k-base_inference.yml'
        # Check model name only if it's one of the standard names (not a local path)
        elif model_name in MODEL_NAMES:
            is_local_path = False
            # Assign config path.
            if model_name == 'evo-1-8k-base' or \
               model_name == 'evo-1-8k-crispr' or \
               model_name == 'evo-1-8k-transposon' or \
               model_name == 'evo-1.5-8k-base':
                config_path = 'configs/evo-1-8k-base_inference.yml'
            elif model_name == 'evo-1-131k-base':
                config_path = 'configs/evo-1-131k-base_inference.yml'
            else:
                raise ValueError(
                    f'Invalid model name {model_name}. Should be one of: '
                    f'{", ".join(MODEL_NAMES)}.'
                )
        else:
            raise ValueError(
                f'Invalid model name or path: {model_name}. Should be one of: '
                f'{", ".join(MODEL_NAMES)} or a valid directory path.'
            )

        # Load model.
        self.model = load_checkpoint(
            model_name=model_name,
            config_path=config_path,
            device=self.device,
            local_files_only=local_files_only,
            is_local_path=is_local_path
        )

        # Load tokenizer.
        self.tokenizer = CharLevelTokenizer(512)

        
HF_MODEL_NAME_MAP = {
    'evo-1.5-8k-base': 'evo-design/evo-1.5-8k-base',
    'evo-1-8k-base': 'togethercomputer/evo-1-8k-base',
    'evo-1-131k-base': 'togethercomputer/evo-1-131k-base',
    'evo-1-8k-crispr': 'LongSafari/evo-1-8k-crispr',
    'evo-1-8k-transposon': 'LongSafari/evo-1-8k-transposon',
}

def load_checkpoint(
    model_name: str = MODEL_NAMES[1],
    config_path: str = 'evo/configs/evo-1-131k-base_inference.yml',
    device: str = None,
    local_files_only: bool = False,
    is_local_path: bool = False,
    *args, **kwargs
):
    """
    Load checkpoint from HuggingFace and place it into SH model.
    
    Args:
        model_name: Name of the model to load or path to a locally downloaded model
        config_path: Path to the config file
        device: Device to load the model on
        local_files_only: If True, only use local files and don't try to download
        is_local_path: If True, model_name is a path to local model files
    """

    # If it's a direct local path, use it
    if is_local_path:
        hf_model_name = model_name
    # Map model name to HuggingFace model name if it's a standard name
    elif model_name in MODEL_NAMES:
        hf_model_name = HF_MODEL_NAME_MAP[model_name]
    else:
        # If it's another kind of path, use it directly
        hf_model_name = model_name
    
    # Print debug information
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', '')
    print(f"Loading model: {hf_model_name}")
    print(f"Using cache directory: {cache_dir}")
    print(f"Local files only mode: {local_files_only}")
    print(f"Is local path: {is_local_path}")

    try:
        # If using a local path and local_files_only is True, 
        # directly load state dict from the pytorch_model.bin file
        if is_local_path and local_files_only:
            # Check for pytorch_model.pt, which is the combined file
            pt_path = os.path.join(hf_model_name, 'pytorch_model.pt')
            if os.path.exists(pt_path):
                print(f"Loading model directly from {pt_path}")
                state_dict = torch.load(pt_path, map_location='cpu')
                
                # Load SH config
                config = yaml.safe_load(pkgutil.get_data(__name__, config_path))
                global_config = dotdict(config, Loader=yaml.FullLoader)
                
                # Load SH Model
                model = StripedHyena(global_config)
                model.load_state_dict(state_dict, strict=True)
                model.to_bfloat16_except_poles_residues()
                if device is not None:
                    model = model.to(device)
                
                return model
        
        # Otherwise proceed with normal loading
        # Load model config.
        model_config = AutoConfig.from_pretrained(
            hf_model_name,
            trust_remote_code=True,
            revision='1.1_fix' if re.match(r'evo-1-.*-base', model_name) else 'main',
            local_files_only=local_files_only,
            cache_dir=cache_dir
        )
        model_config.use_cache = True
    
        # Load model.
        model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            config=model_config,
            trust_remote_code=True,
            revision='1.1_fix' if re.match(r'evo-1-.*-base', model_name) else 'main',
            local_files_only=local_files_only,
            cache_dir=cache_dir
        )
    
        # Load model state dict & cleanup.
        state_dict = model.backbone.state_dict()
        del model
        del model_config
    
        # Load SH config.
        config = yaml.safe_load(pkgutil.get_data(__name__, config_path))
        global_config = dotdict(config, Loader=yaml.FullLoader)
    
        # Load SH Model.
        model = StripedHyena(global_config)
        model.load_state_dict(state_dict, strict=True)
        model.to_bfloat16_except_poles_residues()
        if device is not None:
            model = model.to(device)
    
        return model
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
