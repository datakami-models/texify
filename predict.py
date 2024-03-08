# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import sys
import os
import time
from typing import Optional, Dict
from dataclasses import dataclass, field

WEIGHTS_FOLDER = "/src/models/"
os.environ['HF_HOME'] = WEIGHTS_FOLDER 
os.environ['HF_HUB_CACHE'] = WEIGHTS_FOLDER
os.environ['TORCH_HOME'] = WEIGHTS_FOLDER
os.environ['PYANNOTE_CACHE'] = WEIGHTS_FOLDER

from PIL import Image
import torch

@dataclass
class TexifySettings:
    TORCH_DEVICE: Optional[str] = None
    TORCH_DEVICE_MODEL: str = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_TOKENS: int = 384 # Will not work well above 768, since it was not trained with more
    MAX_IMAGE_SIZE: dict = field(default_factory=lambda: {"height": 420, "width": 420})
    MODEL_CHECKPOINT: str = "vikp/texify"
    BATCH_SIZE: int = 16 # Should use ~5GB of RAM
    DATA_DIR: str = "data"
    TEMPERATURE: float = 0.0 # Temperature for generation, 0.0 means greedy
    MODEL_DTYPE: torch.dtype = torch.float32 if TORCH_DEVICE_MODEL == "cpu" else torch.float16

# monkeypatch computed_field and BaseSettings from texify.settings to fix cog incompatibility with pydantic 2.5.3:
from unittest.mock import Mock

sys.modules['pydantic.computed_field'] = Mock()
sys.modules['pydantic_settings'] = Mock()
sys.modules['pydantic_settings'].BaseSettings = TexifySettings

# import texify
from texify.model.model import load_model
from texify.model.processor import load_processor
from texify.inference import batch_inference
from texify.output import replace_katex_invalid


    
from cog import BasePredictor, Input, Path

    
class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.settings = TexifySettings()
        self.model = load_model(checkpoint=self.settings.MODEL_CHECKPOINT, device=self.settings.TORCH_DEVICE_MODEL, dtype=self.settings.MODEL_DTYPE)
        self.processor = load_processor()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        katex_compatible: bool = Input(description="Make output KaTeX compatible", default=False),
    ) -> str:
        """Run a single prediction on the model"""
        loaded_image = Image.open(image)
        text = batch_inference([loaded_image], self.model, self.processor)
        if katex_compatible:
            text = [replace_katex_invalid(t) for t in text]
        return text[0]
