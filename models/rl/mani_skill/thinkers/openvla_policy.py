from __future__ import annotations
import torch
from dataclasses import dataclass
from PIL import Image
import numpy as np
from transformers import AutoModelForVision2Seq, pipeline, AutoProcessor
from typing import Optional

@dataclass
class OpenVLAActionScaling:
    pos_scale: float = 0.02
    rot_scale: float = 0.10
    grip_scale: float = 1.0



class OpenVLAPolicy:
    """
    OpenVLA wrapper
    """
    def __init__(self, 
                 model_id:str = 'openvla/openvla-7b',
                 device: str = 'cuda:0',
                 use_flash_attn: bool = True,
                 dtype: torch.dtype = torch.bfloat16,
                 unnorm_key: Optional[str] = None,
                 scaling: Optional[OpenVLAActionScaling] = None,):
        self.model_id = model_id
        self.device =device 
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True)
        self.scaler = scaling if scaling is not None else OpenVLAActionScaling
        self.dtype = dtype
        self.unnorm_key = unnorm_key
        kwargs = dict(
            dtype = torch.bfloat16,
            low_cpu_mem_usage = True,
            trust_remote_code = True
        )

        if use_flash_attn:
            kwargs['attn_implementation'] = 'flash_attention_2'

        self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **kwargs)
        self.model.eval()

    def format2Prompt(self, instruction: str):
        return f'In: What action should the robot take to {instruction}?\nOut:'
    
    @torch.no_grad()
    def get_policy(self, observation: np.ndarray, instruction):
        prompt = self.format2Prompt(instruction)
        image = Image.fromarray(observation)
        inputs = self.processor(prompt, image).to(self.device, dtype= self.dtype)
        if self.unnorm_key is not None:
            action = self.model.predict_action(**inputs, unnorm_key = self.unnorm_key, do_sample = False)
        else:
            action = self.model.predict_action(**inputs, unnorm_key = 'bridge_orig', do_sample = False)

        action = action.detach().float().cpu().numpy().reshape(-1)
        assert action.shape[0] == 7, f'expected action dimension {7} but got {action.shape[0]}'
        return action
    
    def scale_policy(self, action):
        a = action.copy()
        a[0:3] = np.clip(a[0:3], -1, 1) * self.scaler.pos_scale
        a[3:6] = np.clip(a[0:3], -1, 1) * self.scaler.rot_scale
        a[-1] = float(np.clip(a[-1], -1, 1)) * self.scaler.grip_scale
        return a
    
    def __call__(self, observation, instruction):
        raw_action = self.get_policy(observation, instruction)
        return self.scale_policy(raw_action)


    