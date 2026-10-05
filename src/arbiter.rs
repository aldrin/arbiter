//! The client: an API key, a model, and one call that answers a batch.

use std::time::Duration;

use serde_json::Value;

use crate::answer::Answers;
use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::question::Request;
use crate::secret::{Secret, SecretChain};

/// How long a request may take before it is abandoned.
///
/// A decision model answers in roughly 100–200 ms, so a request still running
/// after this has stopped being useful on a request path.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(15);

/// A client for one provider's decision endpoint.
///
/// [`Provider::TypeSafe`] unless another is named: see [`Self::for_provider`].
///
/// ```no_run
/// use arbiter::{Arbiter, Provider, Request};
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Arbiter::discover(Provider::TypeSafe)?;
///
/// let answers = client
///     .ask(
///         Request::new("My flight was cancelled. Can I get a refund?")
///             .noul("wants_refund", "Is the customer asking for a refund?")
///             .choice(
///                 "queue",
///                 "Which queue should this go to?",
///                 [("refunds", "Money back"), ("rebooking", "New flight")],
///             ),
///     )
///     .await?;
///
/// if answers.noul("wants_refund")?.is_yes(0.8) {
///     println!("route to {}", answers.choice("queue")?.choice);
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Arbiter {
    http: reqwest::Client,
    provider: Provider,
    base_url: String,
    api_key: Secret,
    model: String,
    timeout: Duration,
}

impl std::fmt::Debug for Arbiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Arbiter")
            .field("provider", &self.provider)
            .field("base_url", &redact_userinfo(&self.base_url))
            .field("model", &self.model)
            .field("api_key", &self.api_key)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

impl Arbiter {
    /// A client on the default provider ([`Provider::TypeSafe`])
    /// authenticating with `api_key`.
    ///
    /// Fails only if the TLS backend cannot be initialised.
    pub fn new(api_key: impl Into<Secret>) -> Result<Self> {
        Self::for_provider(Provider::default(), api_key)
    }

    /// A client on `provider` authenticating with `api_key`, deciding with
    /// that provider's [pinned model](Provider::pinned_model).
    ///
    /// The provider fixes the endpoint and the model vocabulary together, so
    /// it is chosen at construction and not changed afterwards — a client
    /// cannot end up pointed at one provider with the other's base URL.
    ///
    /// Nothing here consults the environment: an embedded tool must not pick
    /// up ambient configuration behind its host's back. Set the model with
    /// [`Self::model`], or pass
    /// [`model_from_env_or`](crate::provider::model_from_env_or) if honouring
    /// `ARBITER_MODEL` is what you want.
    pub fn for_provider(provider: Provider, api_key: impl Into<Secret>) -> Result<Self> {
        Ok(Self {
            http: reqwest::Client::builder().build()?,
            provider,
            base_url: provider.default_base_url().to_owned(),
            api_key: api_key.into(),
            model: provider.pinned_model().to_owned(),
            timeout: DEFAULT_TIMEOUT,
        })
    }

    /// A client on `provider` whose credential comes from the host's secret
    /// sources: the provider's environment variable, then the macOS Keychain.
    ///
    /// See [`secret`](crate::secret) for the chain and how to populate it.
    pub fn discover(provider: Provider) -> Result<Self> {
        Self::with_secrets(provider, &SecretChain::system())
    }

    /// A client on `provider` whose credential comes from the environment
    /// only.
    pub fn from_env(provider: Provider) -> Result<Self> {
        Self::with_secrets(provider, &SecretChain::env_only())
    }

    /// A client on `provider` whose credential comes from `chain`.
    pub fn with_secrets(provider: Provider, chain: &SecretChain) -> Result<Self> {
        Self::for_provider(provider, chain.require(&provider.secret_key())?)
    }

    /// Decide with `model` instead of the default.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Abandon a request that takes longer than `timeout`.
    ///
    /// Applied per request rather than baked into the HTTP client, so this
    /// composes with a client injected through [`Self::http_client`] instead
    /// of silently discarding its configuration.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Send requests to `base_url` instead of the provider's own origin.
    ///
    /// For a proxy, a gateway, or a stub endpoint in tests. The path still
    /// comes from the provider, so this replaces the origin and not the
    /// endpoint.
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        let mut url = base_url.into();
        // Truncating in place avoids reallocating just to drop a slash.
        url.truncate(url.trim_end_matches('/').len());
        self.base_url = url;
        self
    }

    /// Use `http` as the underlying client, for a host that configures its own
    /// connection pool, proxy, or TLS roots.
    #[must_use]
    pub fn http_client(mut self, http: reqwest::Client) -> Self {
        self.http = http;
        self
    }

    /// The model this client decides with.
    pub fn model_id(&self) -> &str {
        &self.model
    }

    /// Who serves this client's requests.
    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// The endpoint requests are posted to.
    pub fn endpoint(&self) -> String {
        format!("{}{}", self.base_url, self.provider.path())
    }

    /// Answer every question in `request`.
    ///
    /// The request is validated before anything is sent, so a malformed batch
    /// costs no credits. Questions are evaluated independently and in
    /// parallel by the provider: a batch of ten is one round trip.
    pub async fn ask(&self, request: Request) -> Result<Answers> {
        request.validate()?;

        tracing::debug!(
            provider = %self.provider,
            model = request.model_override().unwrap_or(&self.model),
            questions = request.len(),
            "asking"
        );

        self.post(request.into_body(&self.model, self.provider))
            .await
    }

    /// Answer every question in `request`, keeping what crossed the wire.
    ///
    /// [`Self::ask`] parses the response and drops it. This keeps both halves,
    /// for a caller that has to show or log what was actually sent and
    /// received — an audit trail, a transcript, a bug report — rather than a
    /// re-serialisation of the parsed answers, which is not the same bytes.
    ///
    /// Costs one clone of the request body over [`Self::ask`], since the body
    /// has to outlive the request it was built from.
    pub async fn ask_recorded(&self, request: Request) -> Result<Exchange> {
        request.validate()?;

        let body = request.into_body(&self.model, self.provider);
        let (answers, response) = self.post_recorded(body.clone()).await?;
        Ok(Exchange {
            request: body,
            response,
            answers,
        })
    }

    /// Post `body` verbatim and parse the answers.
    ///
    /// The escape hatch for a request this crate cannot yet express: `body`
    /// is sent as given, with no validation and no `model` default applied,
    /// so it must already be a complete decision request.
    pub async fn ask_raw(&self, body: Value) -> Result<Answers> {
        self.post(body).await
    }

    /// Send one request body and parse what comes back.
    async fn post(&self, body: Value) -> Result<Answers> {
        self.post_recorded(body)
            .await
            .map(|(answers, _response)| answers)
    }

    /// Send one request body, parsing the response and keeping it.
    async fn post_recorded(&self, body: Value) -> Result<(Answers, String)> {
        let endpoint = self.endpoint();
        tracing::debug!(endpoint = %redact_userinfo(&endpoint), "posting decision request");

        let response = self
            .http
            .post(&endpoint)
            .timeout(self.timeout)
            .bearer_auth(self.api_key.expose())
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let text = response.text().await?;

        if !status.is_success() {
            return Err(Error::Api {
                status: status.as_u16(),
                message: provider_message(&text),
            });
        }

        let answers: Answers = serde_json::from_str(&text).map_err(|source| Error::Decode {
            source,
            raw: text.clone(),
        })?;

        tracing::debug!(
            model = %answers.model,
            answered = answers.len(),
            input_tokens = answers.usage.input_tokens,
            cost = ?answers.usage.cost,
            "answered"
        );

        Ok((answers, text))
    }
}

/// One request and the response it drew, as they crossed the wire.
///
/// From [`Arbiter::ask_recorded`]. The point is fidelity: `request` is the body
/// that was posted and `response` is the body that came back, byte for byte,
/// neither of them reconstructed from [`Self::answers`].
#[derive(Debug, Clone)]
pub struct Exchange {
    /// The body that was posted.
    pub request: Value,
    /// The response body, verbatim.
    pub response: String,
    /// The response, parsed.
    pub answers: Answers,
}

/// How much of an error body is worth carrying into a message.
const MESSAGE_LIMIT: usize = 500;

/// `url` with any `user:password@` userinfo replaced.
///
/// `Secret` keeps the API key out of logs and `Debug`, but a base URL can
/// carry a credential of its own in its userinfo, and the endpoint is both
/// logged and reported by `Debug`. The URL used to make the request is
/// untouched — only what is shown is redacted.
fn redact_userinfo(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_owned();
    };
    // Userinfo, when present, precedes the first `/` of the path.
    let authority_end = rest.find('/').unwrap_or(rest.len());
    // Split at the last `@`, which is the delimiter: an unencoded one inside
    // a password would otherwise end the userinfo early.
    match rest[..authority_end].rsplit_once('@') {
        Some((_, host)) => format!("{scheme}://***@{host}{}", &rest[authority_end..]),
        None => url.to_owned(),
    }
}

/// The provider's complaint out of an error body.
///
/// The two providers disagree on the envelope: OpenRouter documents
/// `{"error": {"code", "message"}}`, while TypeSafe answers
/// `{"detail": {"error_type", "message"}}`. Both are unwrapped, and a gateway
/// or edge returning something else entirely falls back to the raw body so it
/// stays diagnosable rather than being swallowed.
///
/// Every path goes through [`truncate`], because none of this text is trusted:
/// it comes from whatever answered, which with a base-URL override need not be
/// the provider at all.
fn provider_message(body: &str) -> String {
    /// Where a message hides, in the order worth trying.
    const ENVELOPES: [&str; 2] = ["error", "detail"];

    let trimmed = body.trim();
    if trimmed.is_empty() {
        return "empty response body".to_owned();
    }

    serde_json::from_str::<Value>(trimmed)
        .ok()
        .and_then(|value| {
            ENVELOPES
                .iter()
                .filter_map(|envelope| value.get(envelope))
                // A bare string envelope, as in `{"detail": "nope"}`, is as
                // useful as a nested one.
                .find_map(|inner| inner.get("message").unwrap_or(inner).as_str())
                .map(|message| truncate(message, MESSAGE_LIMIT))
        })
        .unwrap_or_else(|| truncate(trimmed, MESSAGE_LIMIT))
}

/// `text`, shortened to `limit` characters and made safe to print.
///
/// An HTML error page from an edge network can be enormous; an error message
/// is read in a terminal. Shortening happens first, so the limit bounds how
/// much of the body is carried regardless of what escaping costs.
fn truncate(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return escape_controls(text);
    }
    let kept: String = text.chars().take(limit).collect();
    format!("{}… ({} bytes total)", escape_controls(&kept), text.len())
}

/// `text` with anything that would steer a terminal rendered inert.
///
/// This string reaches a terminal through the CLI's stderr and it is
/// not trusted — so escape
/// sequences, carriage returns and bidirectional overrides are written as
/// their textual escapes rather than acted on. A hostile endpoint can still
/// say something misleading; it cannot repaint the screen or reorder the line.
fn escape_controls(text: &str) -> String {
    /// Direction overrides and isolates: inert on their own, but they can
    /// visually reorder the text around them.
    const BIDI: [char; 9] = [
        '\u{200e}', '\u{200f}', '\u{202a}', '\u{202b}', '\u{202c}', '\u{202d}', '\u{202e}',
        '\u{2066}', '\u{2069}',
    ];

    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        // `is_control` covers C0 and C1, so the 8-bit CSI is caught too.
        if character.is_control() || BIDI.contains(&character) {
            // `escape_debug` writes ESC as `\u{1b}` and CR as `\r`: readable
            // in a log, and nothing a terminal interprets.
            out.extend(character.escape_debug());
        } else {
            out.push(character);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Arbiter, provider_message, truncate};
    use crate::provider::Provider;

    #[test]
    fn provider_message_prefers_the_documented_envelope() {
        let body = r#"{"error":{"code":402,"message":"Insufficient credits."}}"#;
        assert_eq!(provider_message(body), "Insufficient credits.");
    }

    #[test]
    fn provider_message_unwraps_typesafes_envelope() {
        // Observed from the live endpoint: TypeSafe nests under `detail`,
        // not `error`.
        let body = r#"{"detail":{"error_type":"authentication_error",
                       "message":"Cannot authenticate with the server."}}"#;
        assert_eq!(
            provider_message(body),
            "Cannot authenticate with the server."
        );
    }

    #[test]
    fn provider_message_unwraps_a_bare_string_envelope() {
        assert_eq!(provider_message(r#"{"detail":"not found"}"#), "not found");
    }

    #[test]
    fn provider_message_falls_back_to_the_raw_body() {
        // An edge network returning HTML must stay diagnosable.
        assert_eq!(provider_message("<html>504</html>"), "<html>504</html>");
        assert_eq!(provider_message("   "), "empty response body");
    }

    #[test]
    fn provider_message_falls_back_when_no_envelope_matches() {
        assert_eq!(provider_message(r#"{"wat":"nope"}"#), r#"{"wat":"nope"}"#);
    }

    #[test]
    fn truncate_shortens_only_when_needed() {
        assert_eq!(truncate("short", 10), "short");
        let long = "x".repeat(600);
        let cut = truncate(&long, 500);
        assert!(cut.starts_with(&"x".repeat(500)));
        assert!(cut.contains("600 bytes total"));
    }

    #[test]
    fn a_hostile_body_cannot_steer_the_terminal() {
        // This text reaches a terminal via the CLI's stderr, and it comes
        // from whatever answered the request.
        let body = "\u{1b}[2J\u{1b}[31mnothing is wrong\r";
        let rendered = provider_message(body);
        assert!(!rendered.contains('\u{1b}'), "{rendered}");
        assert!(!rendered.contains('\r'), "{rendered}");
        assert!(rendered.contains("\\u{1b}[2J"), "{rendered}");
        assert!(rendered.contains("nothing is wrong"), "{rendered}");
    }

    #[test]
    fn a_hostile_envelope_message_is_escaped_too() {
        // The documented envelope is no more trustworthy than the raw body.
        // JSON forbids a raw control character in a string, so an escape is
        // the shape a hostile endpoint would actually have to send.
        let body = r#"{"error":{"message":"ok\u001b[31m then"}}"#;
        let rendered = provider_message(body);
        assert!(!rendered.contains('\u{1b}'), "{rendered}");
        assert_eq!(rendered, "ok\\u{1b}[31m then");
    }

    #[test]
    fn bidi_overrides_cannot_reorder_a_message() {
        let rendered = provider_message("quota \u{202e}deliecnac\u{202c}");
        assert!(!rendered.contains('\u{202e}'), "{rendered}");
        assert!(rendered.contains("\\u{202e}"), "{rendered}");
    }

    #[test]
    fn an_over_long_provider_message_is_bounded() {
        let body = format!(r#"{{"error":{{"message":"{}"}}}}"#, "x".repeat(900));
        let rendered = provider_message(&body);
        assert!(rendered.contains("900 bytes total"), "{rendered}");
    }

    #[test]
    fn newlines_survive_as_escapes_rather_than_raw_breaks() {
        // One error, one line: a raw newline could forge a second log entry.
        assert_eq!(
            provider_message("<html>\n504</html>"),
            "<html>\\n504</html>"
        );
    }

    #[test]
    fn a_credential_in_the_base_url_is_kept_out_of_debug() {
        let client = Arbiter::new("k")
            .expect("build")
            .base_url("https://alice:hunter2@gateway.example.com");
        let rendered = format!("{client:?}");
        assert!(!rendered.contains("hunter2"), "{rendered}");
        assert!(!rendered.contains("alice"), "{rendered}");
        assert!(rendered.contains("gateway.example.com"), "{rendered}");

        // The URL actually used must keep the credential: only the shown form
        // is redacted.
        assert!(client.endpoint().contains("hunter2"));
    }

    #[test]
    fn redacting_userinfo_leaves_ordinary_urls_alone() {
        use super::redact_userinfo;
        assert_eq!(
            redact_userinfo("https://api.typesafe.ai/v1/systemone"),
            "https://api.typesafe.ai/v1/systemone"
        );
        assert_eq!(redact_userinfo("not a url"), "not a url");
        // An `@` in the path is not userinfo.
        assert_eq!(redact_userinfo("http://host/a@b"), "http://host/a@b");
        // The last `@` in the authority is the delimiter.
        assert_eq!(
            redact_userinfo("https://u:p@ss@host:9/x"),
            "https://***@host:9/x"
        );
    }

    #[test]
    fn truncate_counts_characters_not_bytes() {
        // Slicing by byte offset here would panic mid-character.
        let text = "é".repeat(600);
        let cut = truncate(&text, 500);
        assert!(cut.contains("1200 bytes total"));
    }

    #[test]
    fn base_url_is_normalised_and_the_endpoint_is_built_from_it() {
        let client = Arbiter::for_provider(Provider::OpenRouter, "k")
            .expect("build")
            .base_url("http://127.0.0.1:9/");
        assert_eq!(client.endpoint(), "http://127.0.0.1:9/api/alpha/decisions");
    }

    #[test]
    fn each_provider_gets_its_own_endpoint_and_default_model() {
        let typesafe = Arbiter::new("k").expect("build");
        assert_eq!(typesafe.provider(), Provider::TypeSafe);
        assert_eq!(typesafe.endpoint(), "https://api.typesafe.ai/v1/systemone");
        assert_eq!(typesafe.model_id(), "jev-1.13.0");

        let openrouter = Arbiter::for_provider(Provider::OpenRouter, "k").expect("build");
        assert_eq!(
            openrouter.endpoint(),
            "https://openrouter.ai/api/alpha/decisions"
        );
        assert_eq!(openrouter.model_id(), "typesafe/jev-1.13");
    }

    #[test]
    fn an_explicit_model_survives_provider_defaults() {
        let client = Arbiter::for_provider(Provider::OpenRouter, "k")
            .expect("build")
            .model("~typesafe/jev-latest");
        assert_eq!(client.model_id(), "~typesafe/jev-latest");
    }

    #[test]
    fn debug_names_the_provider_without_revealing_the_key() {
        let rendered = format!(
            "{:?}",
            Arbiter::for_provider(Provider::OpenRouter, "sk-topsecret").expect("build")
        );
        assert!(rendered.contains("OpenRouter"), "{rendered}");
        assert!(!rendered.contains("topsecret"), "{rendered}");
    }

    #[test]
    fn debug_never_reveals_the_api_key() {
        let rendered = format!("{:?}", Arbiter::new("sk-or-v1-topsecret").expect("build"));
        assert!(!rendered.contains("topsecret"), "{rendered}");
        assert!(rendered.contains("[redacted]"), "{rendered}");
    }
}
