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

impl From<teeny_llm_core::Error> for ApiError {
    fn from(e: teeny_llm_core::Error) -> Self {
        use teeny_llm_core::Error as E;
        let message = e.to_string();
        let (status, kind, param, code) = match &e {
            E::ModelNotFound { .. } => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                Some("model"),
                Some("model_not_found"),
            ),
            E::InvalidRequest(_) => (StatusCode::BAD_REQUEST, "invalid_request_error", None, None),
            E::Unsupported { .. } => (
                StatusCode::NOT_IMPLEMENTED,
                "invalid_request_error",
                None,
                Some("unsupported"),
            ),
            E::Inference(_) => (
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
                param: param.map(str::to_owned),
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
