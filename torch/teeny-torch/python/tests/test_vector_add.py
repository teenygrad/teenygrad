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
