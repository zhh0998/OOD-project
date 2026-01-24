from .base_model import BaseTemporalModel, TimeEncoder
from .dygprompt import DyGPrompt, create_dygprompt
from .tpnet import TPNet, create_tpnet
from .ssm_memory_llm import SSMMemoryLLM, create_ssm_memory_llm

MODEL_REGISTRY = {
    'baseline': BaseTemporalModel,
    'dygprompt': create_dygprompt,
    'tpnet': create_tpnet,
    'ssm_memory_llm': create_ssm_memory_llm,
}

def create_model(model_name, num_nodes, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    model_cls = MODEL_REGISTRY[model_name]
    return model_cls(num_nodes, **kwargs) if model_name == 'baseline' else model_cls(num_nodes, **kwargs)
