"""
models/full_model.py — Full Hybrid Talking-Head Model
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
from config import ModelConfig
from models.keypoint_detector import KeypointDetector
from models.dense_motion import DenseMotionNetwork
from models.generator import SemanticFrameGenerator
from models.fusion import AuxiliaryHEVCStream, FusionModule


class HybridTalkingHeadModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.keypoint_detector    = KeypointDetector(cfg)
        self.dense_motion_network = DenseMotionNetwork(cfg)
        self.semantic_generator   = SemanticFrameGenerator(cfg)
        self.aux_stream           = AuxiliaryHEVCStream(cfg)
        self.fusion_module        = FusionModule(cfg)

    def encode_reference(self, reference_image):
        with torch.no_grad():
            return self.keypoint_detector(reference_image)

    def forward(self, reference_image, driving_frame, hidden_state=None, kp_source=None) -> Dict:
        if kp_source is None:
            kp_source = self.keypoint_detector(reference_image)
        kp_driving = self.keypoint_detector(driving_frame)
        motion_out = self.dense_motion_network(reference_image, kp_source, kp_driving, hidden_state)
        deformation = motion_out['deformation']
        occlusion   = motion_out['occlusion']
        new_hidden  = motion_out['hidden_state']
        gen_out     = self.semantic_generator(reference_image, deformation, occlusion)
        semantic_frame   = gen_out['semantic_frame']
        warped_reference = gen_out['warped_reference']
        aux_frame   = self.aux_stream(driving_frame)
        fused_out   = self.fusion_module(semantic_frame, aux_frame, occlusion)
        return {
            'fused_frame': fused_out['fused_frame'],
            'semantic_frame': semantic_frame,
            'aux_frame': aux_frame,
            'warped_reference': warped_reference,
            'kp_source': kp_source,
            'kp_driving': kp_driving,
            'deformation': deformation,
            'occlusion': occlusion,
            'hidden_state': new_hidden,
        }

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        per_module = {
            name: sum(p.numel() for p in mod.parameters())
            for name, mod in [
                ('keypoint_detector',    self.keypoint_detector),
                ('dense_motion_network', self.dense_motion_network),
                ('semantic_generator',   self.semantic_generator),
                ('aux_stream',           self.aux_stream),
                ('fusion_module',        self.fusion_module),
            ]
        }
        per_module['total'] = total
        return per_module
