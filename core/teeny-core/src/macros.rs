/*
 * Copyright (c) 2026 teenygrad (https://teenygrad.org).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/// Chains a sequence of [`Layer`](crate::nn::Layer)s into a single closure, calling each in
/// order and feeding each output into the next.
#[macro_export]
macro_rules! sequential {
    // Accepts: sequential![a, b, c, ...]
    ( $first:expr $(, $rest:expr )* ) => {
        {
            move |input| {
                let out0 = $first.call(input);
                $(
                    let out = $rest.call(out0);
                    let out0 = out;
                )*
                out0
            }

        }
    };
}
