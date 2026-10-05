//! Finding credentials where the machine keeps them.
//!
//! A [`SecretKey`] names one credential in each store's own vocabulary — an
//! environment variable here, a Keychain service there. A [`SecretSource`] is
//! one place to look, and a [`SecretChain`] is an ordered list of them: the
//! first source holding the credential wins.
//!
//! [`SecretChain::system`] builds the chain appropriate to the host. On macOS
//! that is the environment, then the login Keychain via `/usr/bin/security`.
//! Elsewhere it is the environment alone, and the Keychain source reports
//! itself unavailable rather than failing.
//!
//! The environment is tried first on every platform: it is the cheap override,
//! and it is what CI and containers have.
//!
//! ```no_run
//! use arbiter::secret::{SecretChain, SecretKey};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let secret = SecretChain::system().require(&SecretKey::openrouter())?;
//! # let _ = secret;
//! # Ok(())
//! # }
//! ```
//!
//! # Storing a secret for the Keychain source to find
//!
//! One entry per provider, named for the provider:
//!
//! ```text
//! security add-generic-password -s typesafe   -a "$USER" -w
//! security add-generic-password -s openrouter -a "$USER" -w
//! ```
//!
//! Passing `-w` with no value makes `security` prompt, which keeps the
//! credential out of your shell history and out of the process table.

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{Error, Result};

/// What stands in for a credential wherever one would otherwise be printed.
const REDACTED: &str = "[redacted]";

/// A credential whose `Debug`, `Display` and serialized forms are
/// `[redacted]`.
///
/// Reaching the value takes an explicit [`Secret::expose`], so every place a
/// credential can escape is one grep away.
///
/// ```
/// use arbiter::secret::Secret;
///
/// let secret = Secret::from("sk-or-v1-abc");
/// assert_eq!(format!("{secret:?}"), "[redacted]");
/// assert_eq!(format!("{secret}"), "[redacted]");
/// assert_eq!(secret.expose(), "sk-or-v1-abc");
/// ```
///
/// # What this does not do
///
/// The value is not scrubbed from memory when dropped. Redaction stops a
/// credential being logged by accident; it is not protection against a process
/// that can read your heap.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct Secret(String);

impl Secret {
    /// The credential itself. Every call site is a place a secret can leak.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// Whether no credential was supplied.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl<S: Into<String>> From<S> for Secret {
    fn from(value: S) -> Self {
        Self(value.into())
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(REDACTED)
    }
}

impl std::fmt::Display for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(REDACTED)
    }
}

impl serde::Serialize for Secret {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        serializer.serialize_str(REDACTED)
    }
}

/// macOS ships this binary at a fixed, SIP-protected path. A credential
/// lookup resolves it absolutely rather than through `PATH`, so a hostile
/// `PATH` entry cannot stand in for it.
const SECURITY_BIN: &str = "/usr/bin/security";

/// `security` exits with this when the item simply is not in the keychain,
/// which is a miss to pass along the chain rather than a failure.
const EXIT_ITEM_NOT_FOUND: i32 = 44;

/// Which credential to fetch, named the way each store names it.
///
/// ```
/// use arbiter::secret::SecretKey;
///
/// let key = SecretKey::openrouter();
/// assert_eq!(key.env_var(), "OPENROUTER_API_KEY");
/// assert_eq!(key.service(), "openrouter");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecretKey {
    env_var: String,
    service: String,
    account: Option<String>,
}

impl SecretKey {
    /// A credential held in `env_var` in the environment and under `service`
    /// as a generic password in the Keychain.
    pub fn new(env_var: impl Into<String>, service: impl Into<String>) -> Self {
        Self {
            env_var: env_var.into(),
            service: service.into(),
            account: None,
        }
    }

    /// The TypeSafe API key: `TYPESAFE_API_KEY`, or the `typesafe` generic
    /// password.
    ///
    /// `TYPESAFE_API_KEY` is the variable TypeSafe's own SDKs read, so a
    /// process already configured for them needs nothing further.
    pub fn typesafe() -> Self {
        Self::new("TYPESAFE_API_KEY", "typesafe")
    }

    /// The OpenRouter API key: `OPENROUTER_API_KEY`, or the `openrouter`
    /// generic password.
    pub fn openrouter() -> Self {
        Self::new("OPENROUTER_API_KEY", "openrouter")
    }

    /// Narrow the Keychain lookup to one account.
    ///
    /// Without this, `security` matches on the service alone and returns the
    /// first item it finds — fine when there is one, ambiguous when there are
    /// several.
    #[must_use]
    pub fn account(mut self, account: impl Into<String>) -> Self {
        self.account = Some(account.into());
        self
    }

    /// The environment variable this credential lives in.
    pub fn env_var(&self) -> &str {
        &self.env_var
    }

    /// The Keychain service this credential is stored under.
    pub fn service(&self) -> &str {
        &self.service
    }

    /// The Keychain account the lookup is narrowed to, if any.
    pub fn account_name(&self) -> Option<&str> {
        self.account.as_deref()
    }
}

/// One place a credential might be found.
///
/// A lookup distinguishes three outcomes: found, legitimately absent
/// (`Ok(None)`, so the chain keeps going), and broken (`Err`, which stops the
/// chain — a locked keychain is not the same as an empty one).
pub trait SecretSource: fmt::Debug + Send + Sync {
    /// How this source names itself in logs and errors.
    fn describe(&self) -> String;

    /// Whether this source can be consulted on this host at all.
    ///
    /// An unavailable source is skipped by a [`SecretChain`] instead of
    /// failing it, which is how the Keychain source behaves off macOS.
    fn available(&self) -> bool {
        true
    }

    /// Look the credential up.
    fn lookup(&self, key: &SecretKey) -> Result<Option<Secret>>;
}

/// The process environment.
#[derive(Debug, Clone, Copy, Default)]
pub struct Env;

impl SecretSource for Env {
    fn describe(&self) -> String {
        "environment".to_owned()
    }

    fn lookup(&self, key: &SecretKey) -> Result<Option<Secret>> {
        Ok(present(std::env::var(key.env_var()).ok()))
    }
}

/// Interpret a raw variable value.
///
/// A blank variable counts as unset: exporting an empty value is how a working
/// credential gets accidentally shadowed, and treating it as a miss lets the
/// rest of the chain still find one.
///
/// Pure, so the rule is tested without mutating the process environment.
fn present(value: Option<String>) -> Option<Secret> {
    value
        .filter(|value| !value.trim().is_empty())
        .map(Secret::from)
}

/// The macOS Keychain, read through `/usr/bin/security`.
///
/// Off macOS this reports itself [unavailable](SecretSource::available) and a
/// [`SecretChain`] skips it.
///
/// The credential travels back on the child's stdout, never through argv, so
/// it does not appear in the process table. Only the service and account names
/// are passed as arguments.
///
/// A locked keychain may make `security` prompt for unlock, or fail outright
/// if the process cannot show UI. Either is reported as an error rather than a
/// miss, so a locked keychain never looks like an absent credential.
#[derive(Debug, Clone)]
pub struct Keychain {
    program: PathBuf,
    /// Set when `program` was chosen explicitly, which makes the source
    /// available regardless of the host — this is how it is tested off macOS.
    overridden: bool,
}

impl Default for Keychain {
    fn default() -> Self {
        Self::new()
    }
}

impl Keychain {
    /// The Keychain as read by the system `security` binary.
    pub fn new() -> Self {
        Self {
            program: PathBuf::from(SECURITY_BIN),
            overridden: false,
        }
    }

    /// Read through `program` instead of `/usr/bin/security`.
    ///
    /// Intended for tests, which point it at a stub so the argument handling
    /// and exit-code interpretation can be exercised off macOS.
    #[must_use]
    pub fn with_program(program: impl AsRef<Path>) -> Self {
        Self {
            program: program.as_ref().to_path_buf(),
            overridden: true,
        }
    }

    /// The arguments for a generic-password lookup of `key`.
    fn args(key: &SecretKey) -> Vec<String> {
        let mut args = vec![
            "find-generic-password".to_owned(),
            "-s".to_owned(),
            key.service().to_owned(),
        ];
        if let Some(account) = key.account_name() {
            args.push("-a".to_owned());
            args.push(account.to_owned());
        }
        // `-w` prints the password alone, with no surrounding attributes.
        args.push("-w".to_owned());
        args
    }
}

impl SecretSource for Keychain {
    fn describe(&self) -> String {
        format!("macOS Keychain ({})", self.program.display())
    }

    fn available(&self) -> bool {
        cfg!(target_os = "macos") || self.overridden
    }

    fn lookup(&self, key: &SecretKey) -> Result<Option<Secret>> {
        let output = Command::new(&self.program)
            .args(Self::args(key))
            .output()
            .map_err(|error| {
                Error::secret_source(
                    self.describe(),
                    format!("could not run {}: {error}", self.program.display()),
                )
            })?;

        interpret(&self.describe(), &output)
    }
}

/// Turn a finished `security` run into a lookup outcome.
///
/// Split out from the command so the exit-code and output handling is testable
/// without a keychain.
fn interpret(source: &str, output: &std::process::Output) -> Result<Option<Secret>> {
    if !output.status.success() {
        if output.status.code() == Some(EXIT_ITEM_NOT_FOUND) {
            return Ok(None);
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        let detail = stderr.trim();
        let status = output.status.code().map_or_else(
            || "terminated by signal".to_owned(),
            |code| code.to_string(),
        );
        return Err(Error::secret_source(
            source,
            if detail.is_empty() {
                format!("security exited with {status}")
            } else {
                format!("security exited with {status}: {detail}")
            },
        ));
    }

    let raw = std::str::from_utf8(&output.stdout).map_err(|_| {
        Error::secret_source(
            source,
            "the stored password is not valid UTF-8; `security -w` renders a \
             non-text password as hex, which cannot be used as an API key",
        )
    })?;

    // `security -w` terminates the password with a newline. Strip exactly that
    // terminator rather than trimming, so a credential's own trailing
    // whitespace survives.
    let value = raw
        .strip_suffix('\n')
        .map_or(raw, |line| line.strip_suffix('\r').unwrap_or(line));

    if value.is_empty() {
        return Ok(None);
    }
    Ok(Some(Secret::from(value)))
}

/// An ordered list of places to look for a credential.
///
/// `lookup` returns the first hit; a source that fails stops the search, so a
/// locked keychain is reported rather than silently skipped.
#[derive(Debug)]
pub struct SecretChain {
    sources: Vec<Box<dyn SecretSource>>,
}

impl SecretChain {
    /// An empty chain.
    pub fn empty() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    /// The chain for this host: the environment, then the macOS Keychain.
    ///
    /// Off macOS the Keychain source is present but unavailable, so it is
    /// skipped; the chain is still constructed the same way everywhere, which
    /// keeps behaviour explainable from one code path.
    pub fn system() -> Self {
        Self::empty().with(Env).with(Keychain::new())
    }

    /// The environment only.
    pub fn env_only() -> Self {
        Self::empty().with(Env)
    }

    /// Append a source to consult after the ones already added.
    #[must_use]
    pub fn with(mut self, source: impl SecretSource + 'static) -> Self {
        self.sources.push(Box::new(source));
        self
    }

    /// How the sources in this chain describe themselves, in order.
    pub fn sources(&self) -> Vec<String> {
        self.sources
            .iter()
            .map(|source| source.describe())
            .collect()
    }

    /// Look `key` up in each available source in turn.
    ///
    /// `Ok(None)` means every available source was consulted and none held it.
    pub fn lookup(&self, key: &SecretKey) -> Result<Option<Secret>> {
        for source in &self.sources {
            let described = source.describe();
            if !source.available() {
                tracing::debug!(source = %described, "skipping unavailable secret source");
                continue;
            }
            match source.lookup(key)? {
                Some(secret) => {
                    tracing::debug!(source = %described, key = key.env_var(), "credential found");
                    return Ok(Some(secret));
                }
                None => {
                    tracing::debug!(source = %described, "credential not held here");
                }
            }
        }
        Ok(None)
    }

    /// Look `key` up, failing when no source holds it.
    ///
    /// The error names every source that was actually consulted, so the
    /// remedy is clear from the message alone.
    pub fn require(&self, key: &SecretKey) -> Result<Secret> {
        self.lookup(key)?.ok_or_else(|| Error::SecretNotFound {
            key: key.env_var().to_owned(),
            sources: self
                .sources
                .iter()
                .filter(|source| source.available())
                .map(|source| source.describe())
                .collect(),
        })
    }
}

impl Default for SecretChain {
    fn default() -> Self {
        Self::system()
    }
}

#[cfg(test)]
mod tests {
    use super::Secret;
    use super::{Env, Keychain, SecretChain, SecretKey, SecretSource, interpret, present};
    use crate::error::Error;

    /// A finished process with the given code and streams.
    fn output(code: i32, stdout: &str, stderr: &str) -> std::process::Output {
        use std::os::unix::process::ExitStatusExt as _;
        std::process::Output {
            // `from_raw` takes a wait(2) status: the exit code is the high byte.
            status: std::process::ExitStatus::from_raw(code << 8),
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    #[test]
    fn key_names_the_credential_in_both_vocabularies() {
        let key = SecretKey::openrouter().account("alice");
        assert_eq!(key.env_var(), "OPENROUTER_API_KEY");
        assert_eq!(key.service(), "openrouter");
        assert_eq!(key.account_name(), Some("alice"));
    }

    #[test]
    fn args_request_a_bare_password_for_the_service() {
        let args = Keychain::args(&SecretKey::openrouter());
        assert_eq!(args, ["find-generic-password", "-s", "openrouter", "-w"]);
    }

    #[test]
    fn args_narrow_to_an_account_when_one_is_given() {
        let args = Keychain::args(&SecretKey::openrouter().account("alice"));
        assert_eq!(
            args,
            [
                "find-generic-password",
                "-s",
                "openrouter",
                "-a",
                "alice",
                "-w"
            ]
        );
    }

    #[test]
    fn interpret_strips_only_the_trailing_newline() {
        let secret = interpret("t", &output(0, "sk-or-v1-abc\n", ""))
            .expect("success")
            .expect("a password");
        assert_eq!(secret.expose(), "sk-or-v1-abc");
    }

    #[test]
    fn interpret_strips_a_crlf_terminator() {
        let secret = interpret("t", &output(0, "sk-or-v1-abc\r\n", ""))
            .expect("success")
            .expect("a password");
        assert_eq!(secret.expose(), "sk-or-v1-abc");
    }

    #[test]
    fn interpret_keeps_interior_whitespace() {
        let secret = interpret("t", &output(0, "two words\n", ""))
            .expect("success")
            .expect("a password");
        assert_eq!(secret.expose(), "two words");
    }

    #[test]
    fn interpret_reads_a_missing_item_as_a_miss() {
        let result = interpret("t", &output(44, "", "could not be found")).expect("not an error");
        assert!(result.is_none(), "exit 44 is a miss, not a failure");
    }

    #[test]
    fn interpret_reads_any_other_failure_as_an_error() {
        // A locked keychain must not look like an absent credential.
        match interpret("t", &output(36, "", "interaction not allowed")) {
            Err(Error::SecretSource { message, .. }) => {
                assert!(message.contains("36"), "message: {message}");
                assert!(message.contains("interaction not allowed"));
            }
            other => panic!("expected SecretSource, got {other:?}"),
        }
    }

    #[test]
    fn interpret_treats_an_empty_password_as_a_miss() {
        assert!(
            interpret("t", &output(0, "\n", ""))
                .expect("not an error")
                .is_none()
        );
    }

    #[test]
    fn interpret_rejects_a_non_utf8_password() {
        let mut out = output(0, "", "");
        out.stdout = vec![0xff, 0xfe, b'\n'];
        assert!(matches!(
            interpret("t", &out),
            Err(Error::SecretSource { .. })
        ));
    }

    #[test]
    fn keychain_is_unavailable_off_macos_unless_overridden() {
        assert_eq!(Keychain::new().available(), cfg!(target_os = "macos"));
        assert!(
            Keychain::with_program("/bin/echo").available(),
            "an explicit program makes the source testable anywhere"
        );
    }

    #[test]
    fn a_blank_variable_counts_as_unset() {
        assert!(present(None).is_none());
        assert!(present(Some(String::new())).is_none());
        assert!(present(Some("   \n".to_owned())).is_none());
        assert_eq!(
            present(Some("sk-or-v1-abc".to_owned())).map(|s| s.expose().to_owned()),
            Some("sk-or-v1-abc".to_owned())
        );
    }

    #[test]
    fn env_source_reads_an_absent_variable_as_a_miss() {
        // A name no process would have set, so this needs no mutation of the
        // environment to be deterministic.
        let key = SecretKey::new("ARBITER_ABSENT_VARIABLE_8F3A2B", "unused");
        assert!(Env.lookup(&key).expect("not an error").is_none());
    }

    /// A source that is present but holds nothing.
    #[derive(Debug)]
    struct Empty;
    impl SecretSource for Empty {
        fn describe(&self) -> String {
            "empty".to_owned()
        }
        fn lookup(&self, _: &SecretKey) -> crate::Result<Option<Secret>> {
            Ok(None)
        }
    }

    /// A source that always holds the same value.
    #[derive(Debug)]
    struct Fixed(&'static str);
    impl SecretSource for Fixed {
        fn describe(&self) -> String {
            format!("fixed({})", self.0)
        }
        fn lookup(&self, _: &SecretKey) -> crate::Result<Option<Secret>> {
            Ok(Some(Secret::from(self.0)))
        }
    }

    /// A source that is present but broken.
    #[derive(Debug)]
    struct Broken;
    impl SecretSource for Broken {
        fn describe(&self) -> String {
            "broken".to_owned()
        }
        fn lookup(&self, _: &SecretKey) -> crate::Result<Option<Secret>> {
            Err(Error::secret_source("broken", "locked"))
        }
    }

    /// A source that would panic if consulted, but is never available.
    #[derive(Debug)]
    struct Unavailable;
    impl SecretSource for Unavailable {
        fn describe(&self) -> String {
            "unavailable".to_owned()
        }
        fn available(&self) -> bool {
            false
        }
        fn lookup(&self, _: &SecretKey) -> crate::Result<Option<Secret>> {
            panic!("an unavailable source must never be consulted");
        }
    }

    #[test]
    fn chain_returns_the_first_hit() {
        let chain = SecretChain::empty()
            .with(Empty)
            .with(Fixed("first"))
            .with(Fixed("second"));
        let secret = chain
            .lookup(&SecretKey::openrouter())
            .expect("not an error")
            .expect("a hit");
        assert_eq!(secret.expose(), "first");
    }

    #[test]
    fn chain_skips_unavailable_sources() {
        let chain = SecretChain::empty().with(Unavailable).with(Fixed("env"));
        let secret = chain
            .lookup(&SecretKey::openrouter())
            .expect("not an error")
            .expect("a hit");
        assert_eq!(secret.expose(), "env");
    }

    #[test]
    fn chain_stops_on_a_failing_source() {
        // A broken source must not be papered over by a later hit: a locked
        // keychain is a condition the caller needs to hear about.
        let chain = SecretChain::empty().with(Broken).with(Fixed("later"));
        assert!(matches!(
            chain.lookup(&SecretKey::openrouter()),
            Err(Error::SecretSource { .. })
        ));
    }

    #[test]
    fn require_names_only_the_sources_it_consulted() {
        let chain = SecretChain::empty().with(Unavailable).with(Empty);
        match chain.require(&SecretKey::openrouter()) {
            Err(Error::SecretNotFound { key, sources }) => {
                assert_eq!(key, "OPENROUTER_API_KEY");
                assert_eq!(sources, ["empty"], "an unavailable source is not advice");
            }
            other => panic!("expected SecretNotFound, got {other:?}"),
        }
    }

    #[test]
    fn system_chain_consults_the_environment_before_the_keychain() {
        let sources = SecretChain::system().sources();
        assert_eq!(sources.len(), 2);
        assert_eq!(sources[0], "environment");
        assert!(sources[1].contains("Keychain"), "sources: {sources:?}");
    }

    #[test]
    fn formatting_never_reveals_the_value() {
        let secret = Secret::from("sk-or-v1-topsecret");
        assert_eq!(format!("{secret:?}"), "[redacted]");
        assert_eq!(format!("{secret}"), "[redacted]");
        assert_eq!(
            serde_json::to_string(&secret).expect("serialize"),
            "\"[redacted]\""
        );
    }

    #[test]
    fn nesting_in_a_struct_still_redacts() {
        // The realistic leak: a credential inside a config that gets logged.
        #[derive(Debug, serde::Serialize)]
        struct Config {
            model: &'static str,
            api_key: Secret,
        }

        let config = Config {
            model: "typesafe/jev-1.13",
            api_key: Secret::from("sk-or-v1-topsecret"),
        };

        let debug = format!("{config:?}");
        let json = serde_json::to_string(&config).expect("serialize");
        assert!(!debug.contains("topsecret"), "{debug}");
        assert!(!json.contains("topsecret"), "{json}");
    }

    #[test]
    fn expose_returns_the_value() {
        assert_eq!(Secret::from("abc").expose(), "abc");
        assert!(Secret::default().is_empty());
    }
}
