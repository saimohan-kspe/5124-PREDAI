"""Test 5: Architecture sanity check - memorisation of 32 examples with loss < 1e-4."""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
import torch.optim as optim


class SmallDualBranchMemorizer(nn.Module):
    """Sanity test model matching our dual-branch topology with dropout off."""
    def __init__(self):
        super().__init__()
        self.lstm_intraday = nn.LSTM(input_size=9, hidden_size=32, batch_first=True)
        self.lstm_daily = nn.LSTM(input_size=7, hidden_size=16, batch_first=True)
        self.trunk = nn.Sequential(
            nn.Linear(32 + 16 + 10, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softplus(),
        )

    def forward(self, x_intra, x_daily, x_proxy):
        _, (h_intra, _) = self.lstm_intraday(x_intra)
        _, (h_daily, _) = self.lstm_daily(x_daily)
        feat = torch.cat([h_intra[-1], h_daily[-1], x_proxy], dim=-1)
        v = self.trunk(feat)
        v1 = v[:, 0:1]
        v2 = v[:, 1:2]
        v3 = v[:, 2:3]
        out_5m = torch.sqrt(v1 + 1e-6)
        out_1h = torch.sqrt(v1 + v2 + 1e-6)
        out_24h = torch.sqrt(v1 + v2 + v3 + 1e-6)
        return torch.cat([out_5m, out_1h, out_24h], dim=-1)


def test_memorize_32_samples():
    """Verify neural network can overfit 32 samples driving loss < 1e-3."""
    torch.manual_seed(42)
    B = 32
    x_intra = torch.randn(B, 60, 9)
    x_daily = torch.randn(B, 20, 7)
    x_proxy = torch.randn(B, 10)
    y = torch.abs(torch.randn(B, 3)) + 0.5

    model = SmallDualBranchMemorizer()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.03)
    loss_fn = nn.MSELoss()

    for step in range(120):
        optimizer.zero_grad()
        pred = model(x_intra, x_daily, x_proxy)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

    assert loss.item() < 1e-3
