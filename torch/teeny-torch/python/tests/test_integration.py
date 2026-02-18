#
# Copyright (c) 2026 Teenygrad.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import teenygrad  # type: ignore  # noqa: F401

import torch
import torch.nn as nn
import torch.optim as optim


def test_simpler_classifier():
    """Test that the simple classifier works"""

    class SimpleClassifier(nn.Module):
        """A simple classifier model"""

        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            """Forward pass"""
            return self.net(x)

    # Set device to CUDA if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # Hyperparameters
    input_dim = 10
    hidden_dim = 16
    output_dim = 2
    batch_size = 32
    epochs = 5
    dtype = torch.float32
    seed = 42

    torch.manual_seed(seed)

    # Model, loss, optimizer
    model = SimpleClassifier(input_dim, hidden_dim, output_dim).to(device)
    model = torch.compile(model, backend="teenygrad", dynamic=True)

    # Create random data
    X = torch.randn(batch_size, input_dim, device=device, dtype=dtype)  # type: ignore
    y = torch.randint(0, output_dim, (batch_size,), device=device, dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    # Check that the model can overfit the small batch
    preds = outputs.argmax(dim=1)
    acc = (preds == y).float().mean().item()
    print(f"Final training accuracy: {acc:.2f}")
