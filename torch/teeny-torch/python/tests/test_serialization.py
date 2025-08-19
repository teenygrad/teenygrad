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

import traceback

from teenygrad.graph import deserialize_fx_graph, serialize_fx_graph

import torch
import torch.fx
import torch.nn as nn


def test_serialization():
    """Test that the serialization of an FX graph works"""

    # Create a simple model
    class SimpleModel(torch.nn.Module):
        """A simple model with a linear layer"""
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

        def forward(self, x):
            """Forward pass"""
            return self.net(x)

    # Create the model and trace it
    # Hyperparameters
    input_dim = 10
    hidden_dim = 16
    output_dim = 2

    # Model, loss, optimizer
    model = SimpleModel(input_dim, hidden_dim, output_dim)
    traced = torch.fx.symbolic_trace(model)

    # Create example inputs
    example_inputs = [torch.randn(3, 10)]

    print("Starting serialization...")
    try:
        # Try to serialize
        serialized = serialize_fx_graph(traced, example_inputs)
        assert len(serialized) > 0

        deserialized = deserialize_fx_graph(serialized)
        assert deserialized.NodesLength() == len(traced.graph.nodes)
    except Exception as e:
        print(f"Serialization failed with error: {e}")
        print(f"Error type: {type(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    test_serialization()
