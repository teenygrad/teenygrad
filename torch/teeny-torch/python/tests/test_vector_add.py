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


def test_vector_add():
    """Test that the vector add works"""

    class VectorAdd(nn.Module):
        """A simple vector add."""

        def forward(self, a, b):
            """Forward pass"""
            return a + b

    # Hyperparameters
    INPUT_DIM = 20
    BATCH_SIZE = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Instantiate model, loss, optimizer
    model = VectorAdd().to(device)
    model.compile(backend="teenygrad", dynamic=True)

    a = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)
    b = torch.randn(BATCH_SIZE, INPUT_DIM).to(device)

    # Forward pass
    c = model(a, b)

    # Print the result
    print(c)


if __name__ == "__main__":
    test_vector_add()
