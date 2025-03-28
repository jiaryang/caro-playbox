import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    is_rocm = torch.version.hip is not None
    print("ROCm is available." if is_rocm else "Using NVIDIA CUDA.")
else:
    print("CUDA or ROCm is not available.")
 
model = models.resnet18().to(device)
model.eval()
 
inputs = torch.randn(5, 3, 224, 224).to(device)
 
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("logs"),
    with_stack=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    with record_function("model_inference"):
        with torch.no_grad():
            model(inputs)
 
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
