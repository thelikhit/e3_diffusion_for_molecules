from dataclasses import dataclass

@dataclass
class NoiseScheduleConfig:
    """class to keep tract of Noise Schedule information"""
    scheduler_type: str
    noise_precision: 1e-4
    dataset_info: str
    diffusion_noise_context: str
    input_features: int