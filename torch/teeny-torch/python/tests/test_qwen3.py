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
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

import torch


def test_qwen3_0_6b():
    """Test Qwen3-0.6B model """

    # load the tokenizer and the model
    config = Qwen3Config()
    model = Qwen3ForCausalLM(config)    
    
    model = torch.compile(model, backend="atlas")
    model.forward(torch.randint(0, 10000, (1, 1024)))
