# model_singleton.py
import os
import torch
from models.models import HandWritingSynthesisNet

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'results', 'best_model_synthesis.pt')
_preloaded_model = None

def get_preloaded_model():
    global _preloaded_model
    if _preloaded_model is None:
        model = HandWritingSynthesisNet()
        state_dict = torch.load(_MODEL_PATH, map_location=_device)
        model.load_state_dict(state_dict)
        model.to(_device)
        model.eval()
        _preloaded_model = model
    return _preloaded_model
