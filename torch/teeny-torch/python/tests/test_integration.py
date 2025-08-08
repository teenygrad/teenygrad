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


def test_simple_add():
    """Test that the simple add function works"""
    def add(x, y):
        return x + y
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([3.0, 4.0])
    add = torch.compile(add, backend="atlas")
    result = add(x, y)
    expected = torch.tensor([4.0, 6.0])
    assert torch.allclose(result, expected)
