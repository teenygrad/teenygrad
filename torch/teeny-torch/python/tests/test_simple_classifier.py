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


def test_simple_classifier():
    """Test that the simple classifier works"""

    class SimpleClassifier(nn.Module):
        """A simple classifier with a linear layer, a ReLU activation, and a linear layer."""

        def __init__(self, input_dim=20, hidden_dim=16, output_dim=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, x):
            """Forward pass"""
            return self.net(x)

    # Hyperparameters
    INPUT_DIM = 20
    HIDDEN_DIM = 16
    OUTPUT_DIM = 4
    BATCH_SIZE = 10
    NUM_BATCHES = 5

    # Instantiate model, loss, optimizer
    model = SimpleClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to("cpu")
    model.compile(backend="teenygrad", dynamic=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop with random data
    for batch_idx in range(NUM_BATCHES):
        # Random input and target
        inputs = torch.randn(BATCH_SIZE, INPUT_DIM)
        targets = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,))

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {batch_idx + 1}/{NUM_BATCHES}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_simple_classifier()
