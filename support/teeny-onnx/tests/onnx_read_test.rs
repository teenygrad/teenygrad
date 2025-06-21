/*
 * Copyright \(c\) 2025 Teenygrad. All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

use std::{fs::File, path::PathBuf};

use protobuf::CodedInputStream;
use protobuf::Message;
use teeny_onnx::onnx_proto3::ModelProto;

#[test]
fn test_read_single_relu() {
    let mut resource_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    resource_path.push("onnx/examples/resources/single_relu.onnx");

    let mut file = File::open(resource_path).unwrap();
    let mut stream = CodedInputStream::new(&mut file);
    let model = ModelProto::parse_from(&mut stream).unwrap();

    assert_eq!("backend-test", model.producer_name);
}
