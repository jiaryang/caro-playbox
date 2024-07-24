import torch
from rpdTracerControl import rpdTracerControl
rpdTracerControl.setFilename(name = f"pyt_trace.rpd", append=False)
rpd_profile = rpdTracerControl()

def gemm(A, B):
    """
    General Matrix Multiply (GEMM)
    C = A * B
    """
    return torch.matmul(A, B)

# Example matrices
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to('cuda')
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]]).to('cuda')

# Compute the matrix product
rpd_profile.start()
C = gemm(A, B)
rpd_profile.stop()

print("A:\n", A)
print("B:\n", B)
print("C = A * B:\n", C)

