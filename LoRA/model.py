import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Handwritten LoRA for a linear layer.

    Forward: y = xW^T + b + (alpha/r) * x(BA)^T
    where A in R^{r x in_features}, B in R^{out_features x r}
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias: bool = True,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("r must be a positive integer.")

        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / float(r)

        # Base linear parameters (pretrained weights in real usage).
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # LoRA parameters.
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)

        self.reset_parameters()

        if freeze_base:
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False

    def reset_parameters(self) -> None:
        # Match nn.Linear init style for base parameters.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        # LoRA standard init: A random, B zero -> starts as no-op.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)

        dropped = self.lora_dropout(x)
        lora_hidden = F.linear(dropped, self.lora_A, bias=None)   # (..., r)
        lora_out = F.linear(lora_hidden, self.lora_B, bias=None)   # (..., out_features)

        return base_out + self.scaling * lora_out

    def lora_parameters(self):
        return [self.lora_A, self.lora_B]

    def merged_weight(self) -> torch.Tensor:
        # W_eff = W + (alpha/r) * (B @ A)
        return self.weight + self.scaling * torch.matmul(self.lora_B, self.lora_A)

    @torch.no_grad()
    def merge_weights(self) -> None:
        self.weight.copy_(self.merged_weight())
        nn.init.zeros_(self.lora_B)


if __name__ == "__main__":
    torch.manual_seed(42)

    # Synthetic regression target.
    batch_size = 64
    in_features = 16
    out_features = 8

    x = torch.randn(batch_size, in_features)
    true_w = torch.randn(out_features, in_features)
    true_b = torch.randn(out_features)
    y = F.linear(x, true_w, true_b)

    model = LoRALinear(
        in_features=in_features,
        out_features=out_features,
        r=4,
        alpha=8.0,
        dropout=0.05,
        bias=True,
        freeze_base=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=1e-2)

    print("Trainable parameter count:", sum(p.numel() for p in trainable))

    for step in range(200):
        pred = model(x)
        loss = F.mse_loss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 199:
            print("step={}, loss={:.6f}".format(step, loss.item()))

    with torch.no_grad():
        final_mse = F.mse_loss(model(x), y).item()
    print("Final MSE:", round(final_mse, 6))

    # Optional: merge LoRA into base weight for inference.
    model.merge_weights()
    with torch.no_grad():
        merged_mse = F.mse_loss(model(x), y).item()
    print("MSE after merge:", round(merged_mse, 6))
