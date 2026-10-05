//! The Keychain source driven through a real subprocess.
//!
//! `/usr/bin/security` is replaced with a stub, so the parts that only a
//! subprocess exercises — argument passing, exit codes, stdout capture — are
//! covered on any Unix host rather than only on a Mac with a populated
//! keychain.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};

use arbiter::secret::{Keychain, SecretKey, SecretSource};
use arbiter::{Error, SecretChain};

/// Write an executable shell script named `name` with `body`.
///
/// The script is written and made executable under a temporary name, then
/// renamed into place. Renaming is atomic, so the exec that follows can never
/// see a half-written file or one still held open for writing — the latter
/// fails with `ETXTBSY` and made this flaky when the script was exec'd at the
/// path it was written to. The pid in the name keeps concurrent `cargo test`
/// runs from sharing a path.
fn stub(name: &str, body: &str) -> PathBuf {
    // Cargo hands integration tests a directory of their own for scratch files.
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("keychain-stubs");
    fs::create_dir_all(&dir).expect("create stub dir");

    let path = dir.join(format!("{name}.{}", std::process::id()));
    let partial = path.with_extension("partial");

    fs::write(&partial, format!("#!/bin/sh\n{body}\n")).expect("write stub");
    fs::set_permissions(&partial, fs::Permissions::from_mode(0o755)).expect("chmod stub");
    fs::rename(&partial, &path).expect("publish stub");

    path
}

#[test]
fn reads_a_password_from_the_stub_stdout() {
    let program = stub("found", r"printf 'sk-or-v1-from-keychain\n'");
    let secret = Keychain::with_program(program)
        .lookup(&SecretKey::openrouter())
        .expect("not an error")
        .expect("a password");

    assert_eq!(secret.expose(), "sk-or-v1-from-keychain");
}

#[test]
fn a_credential_never_appears_in_its_debug_output() {
    let program = stub("redacted", r"printf 'sk-or-v1-topsecret\n'");
    let secret = Keychain::with_program(program)
        .lookup(&SecretKey::openrouter())
        .expect("not an error")
        .expect("a password");

    let rendered = format!("{secret:?}");
    assert_eq!(rendered, "[redacted]");
    assert!(
        !rendered.contains("topsecret"),
        "a secret must not be loggable by accident"
    );
}

#[test]
fn passes_the_service_and_account_as_arguments() {
    // The stub echoes its own argv so the invocation can be asserted on.
    let program = stub("echo-args", r#"echo "$@""#);
    let secret = Keychain::with_program(program)
        .lookup(&SecretKey::openrouter().account("alice"))
        .expect("not an error")
        .expect("output");

    assert_eq!(
        secret.expose(),
        "find-generic-password -s openrouter -a alice -w"
    );
}

#[test]
fn exit_44_is_a_miss_not_a_failure() {
    let program = stub(
        "not-found",
        r#"echo "could not be found" >&2
exit 44"#,
    );
    let found = Keychain::with_program(program)
        .lookup(&SecretKey::openrouter())
        .expect("exit 44 must not be an error");

    assert!(found.is_none());
}

#[test]
fn a_locked_keychain_is_an_error_not_a_miss() {
    let program = stub(
        "locked",
        r#"echo "User interaction is not allowed." >&2
exit 36"#,
    );

    match Keychain::with_program(program).lookup(&SecretKey::openrouter()) {
        Err(Error::SecretSource { message, .. }) => {
            assert!(message.contains("36"), "message: {message}");
            assert!(message.contains("User interaction"), "message: {message}");
        }
        other => panic!("expected SecretSource, got {other:?}"),
    }
}

#[test]
fn a_missing_security_binary_is_reported_clearly() {
    let absent = Path::new(env!("CARGO_TARGET_TMPDIR")).join("no-such-security-binary");

    match Keychain::with_program(&absent).lookup(&SecretKey::openrouter()) {
        Err(Error::SecretSource { message, .. }) => {
            assert!(message.contains("could not run"), "message: {message}");
        }
        other => panic!("expected SecretSource, got {other:?}"),
    }
}

#[test]
fn a_chain_falls_through_the_environment_to_the_keychain() {
    // The real environment has no OPENROUTER_API_KEY in the test runner, so
    // the Env source misses and the stubbed Keychain answers. This is the
    // macOS path end to end, without a Mac.
    let program = stub("fallthrough", r"printf 'sk-or-v1-fallthrough\n'");
    let chain = SecretChain::env_only().with(Keychain::with_program(program));

    let secret = chain
        .require(&SecretKey::new(
            "ARBITER_ABSENT_VARIABLE_8F3A2B",
            "openrouter",
        ))
        .expect("the keychain should answer");

    assert_eq!(secret.expose(), "sk-or-v1-fallthrough");
}

#[test]
fn an_exhausted_chain_names_what_it_consulted() {
    let program = stub("empty-chain", "exit 44");
    let chain = SecretChain::env_only().with(Keychain::with_program(program));

    match chain.require(&SecretKey::new(
        "ARBITER_ABSENT_VARIABLE_8F3A2B",
        "openrouter",
    )) {
        Err(Error::SecretNotFound { key, sources }) => {
            assert_eq!(key, "ARBITER_ABSENT_VARIABLE_8F3A2B");
            assert_eq!(sources.len(), 2, "sources: {sources:?}");
            assert_eq!(sources[0], "environment");
            assert!(sources[1].contains("Keychain"), "sources: {sources:?}");
        }
        other => panic!("expected SecretNotFound, got {other:?}"),
    }
}
