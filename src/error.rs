//! The error type every fallible operation in this crate returns.

/// What went wrong.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The request could not be posed as stated. Raised locally, before
    /// anything is sent.
    #[error("malformed request: {0}")]
    Request(String),

    /// The request never completed: a transport, TLS, or timeout failure.
    #[error("could not reach the decision endpoint: {0}")]
    Transport(#[from] reqwest::Error),

    /// The API refused the request. `message` is the provider's own, which is
    /// usually the actionable half.
    #[error("the decision endpoint returned {status}: {message}")]
    Api {
        /// The HTTP status.
        status: u16,
        /// The provider's message, or the raw body when it was not the
        /// documented error envelope.
        message: String,
    },

    /// The response was not the documented shape.
    #[error("could not parse the decision response: {source}")]
    Decode {
        /// The deserializer's complaint.
        source: serde_json::Error,
        /// What came back, for logs and bug reports.
        raw: String,
    },

    /// No answer came back under this name.
    #[error("no answer named {name:?}; answered: {}", answered.join(", "))]
    NoSuchAnswer {
        /// The name that was asked for.
        name: String,
        /// The names that were answered.
        answered: Vec<String>,
    },

    /// An answer came back as a different primitive than the one requested.
    #[error("answer {name:?} is a {got}, not a {wanted}")]
    AnswerType {
        /// The question's name.
        name: String,
        /// The primitive that was asked for.
        wanted: &'static str,
        /// The primitive that came back.
        got: &'static str,
    },

    /// No secret source held the credential.
    #[error(
        "no credential for {key}: looked in {}",
        if sources.is_empty() { "no available source".to_owned() } else { sources.join(", ") }
    )]
    SecretNotFound {
        /// The credential that was wanted, by its environment-variable name.
        key: String,
        /// The sources actually consulted, in order.
        sources: Vec<String>,
    },

    /// A secret source could not be consulted. Distinct from a credential
    /// being absent: a locked keychain is not an empty one.
    #[error("{source_name}: {message}")]
    SecretSource {
        /// The source that failed, as it describes itself.
        source_name: String,
        /// What went wrong.
        message: String,
    },
}

impl Error {
    /// An [`Error::Request`] naming what was wrong.
    pub(crate) fn request(msg: impl Into<String>) -> Self {
        Self::Request(msg.into())
    }

    /// An [`Error::SecretSource`] naming the source that failed.
    pub(crate) fn secret_source(
        source_name: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::SecretSource {
            source_name: source_name.into(),
            message: message.into(),
        }
    }

    /// Whether retrying the identical request could plausibly succeed.
    ///
    /// True for the transient server-side conditions the API documents —
    /// rate limiting, an overloaded or unavailable provider, a gateway or
    /// edge timeout — and for transport failures. False for anything caused
    /// by the request itself, which will fail the same way every time.
    pub fn is_transient(&self) -> bool {
        match self {
            Self::Transport(error) => !error.is_builder(),
            Self::Api { status, .. } => {
                matches!(
                    status,
                    408 | 409 | 425 | 429 | 500 | 502 | 503 | 504 | 524 | 529
                )
            }
            _ => false,
        }
    }
}

/// A decision result.
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::Error;

    fn api(status: u16) -> Error {
        Error::Api {
            status,
            message: "x".into(),
        }
    }

    #[test]
    fn transient_covers_the_documented_retryable_statuses() {
        for status in [429, 500, 502, 503, 524, 529] {
            assert!(api(status).is_transient(), "{status} should be transient");
        }
    }

    #[test]
    fn a_caller_error_is_never_transient() {
        // Retrying these just spends another request to fail identically.
        for status in [400, 401, 402, 403, 404, 413] {
            assert!(!api(status).is_transient(), "{status} should be permanent");
        }
        assert!(!Error::request("no questions").is_transient());
    }
}
