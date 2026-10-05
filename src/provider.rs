//! Who serves the decision: TypeSafe directly, or OpenRouter.
//!
//! The two differ in more than a hostname — the path, the credential, the
//! model names, and which request fields are accepted all move together. A
//! [`Provider`] is the one place that knows all of it, so a client is
//! configured by naming one rather than by setting four things consistently.
//!
//! | | [`TypeSafe`](Provider::TypeSafe) | [`OpenRouter`](Provider::OpenRouter) |
//! |---|---|---|
//! | Endpoint | `api.typesafe.ai/v1/systemone` | `openrouter.ai/api/alpha/decisions` |
//! | Credential | `TYPESAFE_API_KEY`, Keychain `typesafe` | `OPENROUTER_API_KEY`, Keychain `openrouter` |
//! | Pinned model | `jev-1.13.0` | `typesafe/jev-1.13` |
//! | Latest alias | `jev-latest` | `~typesafe/jev-latest` |
//! | Extra request fields | none | `session_id`, `user` |
//!
//! Both speak the same request body and the same answers, and TypeSafe's
//! response is a subset of OpenRouter's, so nothing downstream of the
//! transport cares which one answered.

use crate::secret::SecretKey;

/// Where a decision request is sent.
///
/// [`TypeSafe`](Self::TypeSafe) is the default: it is the shortest path to the
/// model, and the one whose price and rate limits are published against the
/// model itself.
///
/// ```
/// use arbiter::Provider;
///
/// assert_eq!(Provider::default(), Provider::TypeSafe);
/// assert_eq!(Provider::TypeSafe.pinned_model(), "jev-1.13.0");
/// assert_eq!(Provider::OpenRouter.pinned_model(), "typesafe/jev-1.13");
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Provider {
    /// TypeSafe's own API, `POST /v1/systemone`.
    #[default]
    TypeSafe,
    /// OpenRouter's Decisions API, `POST /api/alpha/decisions`, billed to an
    /// OpenRouter account.
    OpenRouter,
}

impl Provider {
    /// Every provider, for iterating in tests and help text.
    pub const ALL: [Self; 2] = [Self::TypeSafe, Self::OpenRouter];

    /// The short name, as a flag or a config file spells it.
    pub fn name(self) -> &'static str {
        match self {
            Self::TypeSafe => "typesafe",
            Self::OpenRouter => "openrouter",
        }
    }

    /// The origin requests go to, absent an override.
    pub fn default_base_url(self) -> &'static str {
        match self {
            Self::TypeSafe => "https://api.typesafe.ai",
            Self::OpenRouter => "https://openrouter.ai",
        }
    }

    /// The path a request is posted to.
    pub fn path(self) -> &'static str {
        match self {
            Self::TypeSafe => "/v1/systemone",
            Self::OpenRouter => "/api/alpha/decisions",
        }
    }

    /// The credential this provider authenticates with, in each store's own
    /// vocabulary.
    pub fn secret_key(self) -> SecretKey {
        match self {
            Self::TypeSafe => SecretKey::typesafe(),
            Self::OpenRouter => SecretKey::openrouter(),
        }
    }

    /// The versioned model used when the caller names none.
    ///
    /// Pinned rather than an alias, so a confidence threshold tuned against
    /// the default keeps meaning what it meant. Ask for
    /// [`latest_model`](Self::latest_model) to opt into drift.
    pub fn pinned_model(self) -> &'static str {
        match self {
            Self::TypeSafe => "jev-1.13.0",
            Self::OpenRouter => "typesafe/jev-1.13",
        }
    }

    /// The alias that always resolves to the provider's newest model.
    pub fn latest_model(self) -> &'static str {
        match self {
            Self::TypeSafe => "jev-latest",
            // The tilde is part of the slug, not decoration.
            Self::OpenRouter => "~typesafe/jev-latest",
        }
    }

    /// The environment variable a host may read a base-URL override from.
    ///
    /// Nothing in this crate reads it; it is named here so a binary can
    /// honour the same variable the vendor's own SDKs do. `TYPESAFE_BASE_URL`
    /// is what TypeSafe's SDKs read; `OPENROUTER_BASE_URL` follows the same
    /// convention, so either provider can be pointed at a gateway, a staging
    /// endpoint, or a stub without a code change.
    pub fn base_url_env(self) -> &'static str {
        match self {
            Self::TypeSafe => "TYPESAFE_BASE_URL",
            Self::OpenRouter => "OPENROUTER_BASE_URL",
        }
    }

    /// Whether this provider accepts the `session_id` and `user` request
    /// fields.
    ///
    /// They are OpenRouter's observability extensions. TypeSafe documents the
    /// body as `state`, `model` and `questions` only, so sending them there
    /// risks a validation failure for no benefit.
    pub fn accepts_attribution(self) -> bool {
        matches!(self, Self::OpenRouter)
    }
}

/// The environment variable a host may override the model with.
pub const MODEL_ENV: &str = "ARBITER_MODEL";

/// The model named by [`MODEL_ENV`], else `fallback`.
///
/// Opt-in: no arbiter consults this on its own. A tool embedded in someone
/// else's process must not quietly reconfigure itself from that process's
/// environment, so honouring `ARBITER_MODEL` is the host's decision.
///
/// The value is not validated against the provider. A slug for the wrong
/// provider will be rejected by that provider, not here.
///
/// ```
/// use arbiter::provider::{self, Provider};
///
/// let chosen = provider::model_from_env_or(Provider::TypeSafe.pinned_model());
/// assert!(!chosen.is_empty());
/// ```
pub fn model_from_env_or(fallback: &str) -> String {
    std::env::var(MODEL_ENV)
        .ok()
        .map(|model| model.trim().to_owned())
        .filter(|model| !model.is_empty())
        .unwrap_or_else(|| fallback.to_owned())
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Provider {
    type Err = UnknownProvider;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        // Lenient on separators and case, since this arrives from a flag, an
        // environment variable, or a config file.
        let normalised: String = name
            .trim()
            .to_ascii_lowercase()
            .chars()
            .filter(char::is_ascii_alphanumeric)
            .collect();

        match normalised.as_str() {
            "typesafe" | "ts" => Ok(Self::TypeSafe),
            "openrouter" | "or" => Ok(Self::OpenRouter),
            _ => Err(UnknownProvider {
                given: name.to_owned(),
            }),
        }
    }
}

/// A name that does not match any [`Provider`].
#[derive(Debug, Clone, thiserror::Error)]
#[error("unknown provider {given:?}; expected one of: typesafe, openrouter")]
pub struct UnknownProvider {
    /// What was given.
    pub given: String,
}

#[cfg(test)]
mod tests {
    use super::Provider;

    #[test]
    fn typesafe_is_the_default() {
        assert_eq!(Provider::default(), Provider::TypeSafe);
    }

    #[test]
    fn each_provider_carries_a_distinct_endpoint_and_credential() {
        // The point of the type: these cannot be configured inconsistently.
        let [typesafe, openrouter] = Provider::ALL;

        assert_eq!(typesafe.default_base_url(), "https://api.typesafe.ai");
        assert_eq!(typesafe.path(), "/v1/systemone");
        assert_eq!(typesafe.secret_key().env_var(), "TYPESAFE_API_KEY");
        assert_eq!(typesafe.secret_key().service(), "typesafe");

        assert_eq!(openrouter.default_base_url(), "https://openrouter.ai");
        assert_eq!(openrouter.path(), "/api/alpha/decisions");
        assert_eq!(openrouter.secret_key().env_var(), "OPENROUTER_API_KEY");
        assert_eq!(openrouter.secret_key().service(), "openrouter");
    }

    #[test]
    fn the_latest_alias_keeps_its_tilde_only_on_openrouter() {
        // The tilde is part of the slug, not decoration.
        assert!(Provider::OpenRouter.latest_model().starts_with('~'));
        assert!(!Provider::TypeSafe.latest_model().starts_with('~'));
    }

    #[test]
    fn typesafe_slugs_carry_no_vendor_prefix() {
        for slug in [
            Provider::TypeSafe.pinned_model(),
            Provider::TypeSafe.latest_model(),
        ] {
            assert!(!slug.contains('/'), "{slug}");
        }
    }

    #[test]
    fn the_model_fallback_is_used_when_the_variable_is_unset() {
        // The test runner has no ARBITER_MODEL, so this exercises the
        // fallback path without mutating the environment.
        assert_eq!(super::model_from_env_or("jev-1.13.0"), "jev-1.13.0");
    }

    #[test]
    fn model_names_differ_between_providers() {
        // A slug from one provider is not valid at the other, which is why
        // the default model follows the provider.
        assert_eq!(Provider::TypeSafe.pinned_model(), "jev-1.13.0");
        assert_eq!(Provider::TypeSafe.latest_model(), "jev-latest");
        assert_eq!(Provider::OpenRouter.pinned_model(), "typesafe/jev-1.13");
        assert_eq!(Provider::OpenRouter.latest_model(), "~typesafe/jev-latest");
    }

    #[test]
    fn defaults_are_pinned_not_floating() {
        for provider in Provider::ALL {
            assert_ne!(
                provider.pinned_model(),
                provider.latest_model(),
                "{provider} must default to a pinned version"
            );
        }
    }

    #[test]
    fn both_providers_can_be_pointed_elsewhere() {
        assert_eq!(Provider::TypeSafe.base_url_env(), "TYPESAFE_BASE_URL");
        assert_eq!(Provider::OpenRouter.base_url_env(), "OPENROUTER_BASE_URL");
    }

    #[test]
    fn only_openrouter_accepts_attribution_fields() {
        assert!(!Provider::TypeSafe.accepts_attribution());
        assert!(Provider::OpenRouter.accepts_attribution());
    }

    #[test]
    fn parsing_is_lenient_about_case_and_separators() {
        for name in ["typesafe", "TypeSafe", "type-safe", " TYPE_SAFE ", "ts"] {
            assert_eq!(
                name.parse::<Provider>().expect(name),
                Provider::TypeSafe,
                "{name}"
            );
        }
        for name in [
            "openrouter",
            "OpenRouter",
            "open-router",
            "open_router",
            "or",
        ] {
            assert_eq!(
                name.parse::<Provider>().expect(name),
                Provider::OpenRouter,
                "{name}"
            );
        }
    }

    #[test]
    fn an_unknown_name_says_what_was_expected() {
        let error = "anthropic".parse::<Provider>().expect_err("not a provider");
        let message = error.to_string();
        assert!(message.contains("anthropic"), "{message}");
        assert!(message.contains("typesafe"), "{message}");
        assert!(message.contains("openrouter"), "{message}");
    }

    #[test]
    fn names_round_trip_through_display() {
        for provider in Provider::ALL {
            assert_eq!(
                provider
                    .to_string()
                    .parse::<Provider>()
                    .expect("round trip"),
                provider
            );
        }
    }
}
