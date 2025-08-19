#
# Copyright (c) 2025 Teenygrad. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
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
                nn.Linear(hidden_dim, output_dim)
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
    model = SimpleClassifier(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM).to('cpu')
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

        print(f"Batch {batch_idx+1}/{NUM_BATCHES}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_simple_classifier()
