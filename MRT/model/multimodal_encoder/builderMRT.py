from .clip_encoderMRT import CLIPVisionTower


def build_vision_tower(vision_tower_cfg,V_rank, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    return CLIPVisionTower(vision_tower, args=vision_tower_cfg, V_rank=V_rank, **kwargs)

    
