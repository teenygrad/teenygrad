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

use axum::{
    Json,
    extract::rejection::JsonRejection,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};

/// `ErrorResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

/// `Error`. `param` and `code` are required-but-nullable in the spec, so they're always emitted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub param: Option<String>,
    pub code: Option<String>,
}

/// An error with the HTTP status it should be served with.
#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: StatusCode,
    pub body: ErrorBody,
}

impl ApiError {
    pub fn invalid_request(message: impl Into<String>, param: Option<&str>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorBody {
                message: message.into(),
                kind: "invalid_request_error".to_owned(),
                param: param.map(str::to_owned),
                code: None,
            },
        }
    }

    pub fn into_response_body(self) -> ErrorResponse {
        ErrorResponse { error: self.body }
    }
}

#[cfg(feature = "llm")]
impl From<teeny_llm::Error> for ApiError {
    fn from(e: teeny_llm::Error) -> Self {
        crate::Error::from(e).into()
    }
}

impl From<crate::Error> for ApiError {
    fn from(e: crate::Error) -> Self {
        use crate::Error as E;
        let message = e.to_string();
        let (status, kind, param, code) = match &e {
            E::ModelNotFound { .. } => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                Some("model".to_owned()),
                Some("model_not_found"),
            ),
            E::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error", None, None),
            E::InvalidParameter { parameter, .. } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                Some(parameter.clone()),
                Some("invalid_value"),
            ),
            E::UnsupportedParameter { parameter, .. } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                Some(parameter.clone()),
                Some("unsupported_parameter"),
            ),
            E::ModelLacksApi { .. } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                Some("model".to_owned()),
                Some("model_not_supported"),
            ),
            E::Unsupported { .. } => (
                StatusCode::NOT_IMPLEMENTED,
                "invalid_request_error",
                None,
                Some("unsupported"),
            ),
            E::Model(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                None,
                None,
            ),
        };
        Self {
            status,
            body: ErrorBody {
                message,
                kind: kind.to_owned(),
                param,
                code: code.map(str::to_owned),
            },
        }
    }
}

impl From<JsonRejection> for ApiError {
    fn from(r: JsonRejection) -> Self {
        let mut e = Self::invalid_request(r.body_text(), None);
        e.status = r.status();
        e
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.into_response_body())).into_response()
    }
}
