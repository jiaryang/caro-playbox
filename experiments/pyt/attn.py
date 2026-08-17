import torch
import torch.nn.functional as F

def stable_softmax(x, dim=-1):
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    softmax = x_exp / x_exp.sum(dim=dim, keepdim=True)

    print("max(x):\n", x_max)
    print("exp(x - max):\n", x_exp)
    print("softmax result:\n", softmax)
    return softmax

def test_attention_debug_with_custom_v():
    Q = torch.tensor([[1.], [2.], [3.], [4.]])
    K = torch.tensor([[1.], [1.], [0.], [0.]])
    V = torch.tensor([[5.], [10.], [15.], [20.]])

    print(f"{Q=}")
    print(f"{K=}")
    print(f"{V=}")

    d_k = Q.shape[1]
    scale = 1.0 / torch.sqrt(torch.tensor(float(d_k)))

    scores = Q @ K.T
    print("QK^T:\n", scores)

    scaled_scores = scores * scale
    print("Scaled QK^T (before softmax):\n", scaled_scores)

    attn_weights = stable_softmax(scaled_scores, dim=-1)

    output = attn_weights @ V
    print("\n[Final Output]")
    print("Attention output:\n", output)

test_attention_debug_with_custom_v()

