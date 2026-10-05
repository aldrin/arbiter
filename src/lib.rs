//! An arbiter: structured decisions from a model, for code to act on.
//!
//! This is not a client for a language model. A *decision model* does not
//! generate text — it returns a typed answer with calibrated probabilities,
//! drawn from the options you supplied. There is no prose to parse, no
//! prompt to tune, and no chat completion involved; the endpoint is its own.
//!
//! The practical difference: a generative model hands you a paragraph and
//! leaves you to extract a verdict. An arbiter hands you the verdict, with a
//! number saying how sure it is.
//!
//! Send one `state` and a battery of named questions. Each is evaluated
//! independently and in parallel, so a batch of ten is one round trip.
//!
//! # A tool, not an agent
//!
//! This crate is one `await` you call from wherever your decisions are made —
//! a request handler, a batch job, or a step inside somebody else's agent
//! loop. It owns no loop, keeps no conversation, holds no state between calls,
//! and installs nothing global. [`Arbiter::ask`] is the entire surface.
//!
//! Consequences worth relying on:
//!
//! - **No ambient configuration.** [`Arbiter::new`] reads nothing from the
//!   environment; the host passes the key and the model. The
//!   [`provider::model_from_env_or`] and [`Arbiter::discover`] helpers exist for hosts
//!   that *want* that, and say so in their names.
//! - **No runtime of its own.** The library never names `tokio`; it borrows
//!   whichever executor polls it. Supply your own
//!   [`reqwest::Client`](Arbiter::http_client) to share a connection pool, proxy,
//!   or TLS roots.
//! - **Nothing global.** No subscriber is installed, no logger configured.
//!   Tracing spans are emitted and ignored unless the host collects them.
//! - **Default features are empty.** A library consumer inherits no argument
//!   parser and no async runtime. The binary is behind `--features cli`.
//! - **Cheap to clone.** [`Arbiter`] is `Clone` and shares its connection pool,
//!   so one per process is fine and so is one per task.
//!
//! # Three primitives
//!
//! | Primitive | Asks | Answered by |
//! |---|---|---|
//! | [`noul`](Request::noul) | a yes/no judgement | a probability |
//! | [`choice`](Request::choice) | which of these options | one option, plus a distribution |
//! | [`score`](Request::score) | where on this ordered scale | a point on the scale |
//!
//! # Example
//!
//! ```no_run
//! use arbiter::{Arbiter, Provider, Request};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Arbiter::discover(Provider::TypeSafe)?;
//!
//! let answers = client
//!     .ask(
//!         Request::new(serde_json::json!({
//!             "ticket": "My checkout page shows a blank screen after I click Pay.",
//!             "customer_tier": "enterprise",
//!         }))
//!         .noul_described(
//!             "is_bug",
//!             "Is the customer reporting a software defect?",
//!             "The customer describes broken or unexpected product behavior.",
//!             "The customer is asking a question or requesting a feature.",
//!         )
//!         .choice(
//!             "team",
//!             "Which team should own this ticket?",
//!             [
//!                 ("payments", "Checkout, billing, or payment processing issues."),
//!                 ("frontend", "Rendering, layout, or browser compatibility issues."),
//!                 ("account", "Login, permissions, or profile issues."),
//!             ],
//!         )
//!         .score(
//!             "urgency",
//!             "How urgent is this ticket?",
//!             [
//!                 "Can wait for the next release",
//!                 "Should be fixed this week",
//!                 "Blocking revenue right now",
//!             ],
//!         ),
//!     )
//!     .await?;
//!
//! // Code stays in control: combine the independent signals yourself.
//! let team = answers.choice("team")?;
//! if answers.noul("is_bug")?.is_yes(0.8) && team.is_confident(0.75) {
//!     route_to(&team.choice, answers.score("urgency")?.nearest_level());
//! } else {
//!     escalate_to_human();
//! }
//! # fn route_to(_: &str, _: usize) {}
//! # fn escalate_to_human() {}
//! # Ok(())
//! # }
//! ```
//!
//! # Designing questions
//!
//! Ask narrow, atomic questions and combine them in code. One broad question
//! hides several judgements behind a single answer; several specific ones
//! expose each judgement so you can inspect and tune them. Point a question at
//! a field with backticks — ``"Does `ticket.body` request a refund?"`` — and
//! send only the state the questions need.
//!
//! Because decomposition is free, prefer more questions to vaguer ones.
//!
//! # Acting on confidence
//!
//! Confidence is a calibrated probability, not a model's self-report, so a
//! threshold on it means something. Gate on it: act when confident, escalate
//! to a person or a reasoning model when not. [`Choice::is_confident`] fails
//! closed when the provider reports no confidence at all.
//!
//! # Two providers
//!
//! The same model is reachable two ways, and they differ in more than a
//! hostname: the path, the credential, the model names, and which request
//! fields are accepted all move together. A [`Provider`] is the one place
//! that knows all of it, so picking one configures the rest consistently.
//!
//! ```no_run
//! use arbiter::{Arbiter, Provider};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // TypeSafe directly — the default.
//! let direct = Arbiter::discover(Provider::TypeSafe)?;
//!
//! // Or through OpenRouter, billed to an OpenRouter account.
//! let routed = Arbiter::discover(Provider::OpenRouter)?;
//! # let _ = (direct, routed);
//! # Ok(())
//! # }
//! ```
//!
//! Model slugs are **not** interchangeable: `jev-1.13.0` at TypeSafe is
//! `typesafe/jev-1.13` at OpenRouter. Prefer
//! [`Provider::pinned_model`] over naming one by hand. See [`provider`] for
//! the full comparison.
//!
//! # Credentials
//!
//! Pass one in with [`Arbiter::new`], which touches nothing ambient. For hosts
//! that would rather have it found for them, [`Arbiter::discover`] resolves the
//! provider's own credential — `TYPESAFE_API_KEY` or `OPENROUTER_API_KEY`,
//! then on macOS the `typesafe` or `openrouter` Keychain entry. See
//! [`secret`] for the chain and how to extend it.
//!
//! # What you do not get
//!
//! No rationale, reasoning, or explanation — a decision model returns typed
//! answers and probabilities and nothing else. That is the trade: you give up
//! the narration and get an answer your code can branch on. When you need
//! prose, use a generative model.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod answer;
pub mod arbiter;
pub mod error;
pub mod eval;
pub mod provider;
pub mod question;
pub mod secret;

pub use answer::{Answer, Answers, Choice, Noul, Score, Usage};
pub use arbiter::{Arbiter, Exchange};
pub use error::{Error, Result};
pub use provider::Provider;
pub use question::{Guidance, NoulCriteria, Question, Request, SINGLE_QUESTION_NAME};
pub use secret::{Secret, SecretChain, SecretKey, SecretSource};
