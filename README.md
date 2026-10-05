# arbiter

A Rust client for a *decision model* — one that returns a typed answer with
calibrated probabilities instead of prose. You send some state and a battery of
named questions; you get back a verdict per question, with a number saying how
sure it is. Nothing to parse, no prompt to tune.

Three question types:

| Type | Asks | Answered by |
|---|---|---|
| `noul` | a yes/no judgement | a probability (`0.96`) |
| `choice` | which of these options | one option, plus a distribution over all of them |
| `score` | where on this ordered scale | a point on the scale (`1.99`) and a legend |

The repository is the library, a CLI driver, and one worked example that
applies it to a known set of questions — a harness that asks them, and a
renderer that draws the run.

## Library

```rust
use arbiter::{Arbiter, Provider, Request};

let arbiter = Arbiter::discover(Provider::TypeSafe)?;

let answers = arbiter.ask(
    Request::new(serde_json::json!({
        "ticket": "Checkout shows a blank screen after I click Pay."
    }))
    .noul("is_bug", "Does `ticket` describe a software defect?")
    .choice("team", "Which team should own `ticket`?", [
        ("payments", "Checkout, billing, and payment processing."),
        ("frontend", "Rendering, layout, and browser compatibility."),
    ])
    .score("urgency", "How urgent is `ticket`?", [
        "Can wait for the next release",
        "Needs a fix this week",
        "Blocks revenue now",
    ]),
).await?;

let team = answers.choice("team")?;
if answers.noul("is_bug")?.is_yes(0.8) && team.is_confident(0.75) {
    println!("route to {}", team.choice);
}
```

Three questions, **one round trip**. A request carries one `state` and any
number of questions about it, and the provider evaluates them independently and
in parallel — so decomposing a broad judgement into narrow ones is free, and
batching is the way the API is meant to be used.

The corollary is that a batch is questions sharing *one* state. Questions with
nothing in common do not belong in the same request: every question sees the
whole state, so bundling unrelated ones both wastes tokens and lets one
question's context bleed into another's judgement. Ask specific questions,
describe each option, and send only the fields those questions need — a
question can point at one with backticks, as above.

`Arbiter::ask` is the whole surface. Two variants exist for when it isn't
enough: `ask_recorded` returns the request and response bytes alongside the
parsed answers, for an audit trail or a transcript; `ask_raw` posts a body
verbatim, for a request the builder cannot yet express.

Typed accessors error rather than defaulting when a name is missing or came
back as a different type. `is_confident` returns false when the provider
reported no confidence, so the gate fails closed. `Choice::margin` says how far
ahead the winner was, which confidence alone does not always reveal.

The crate enables no default features, installs nothing global, and never names
`tokio` — the host supplies the runtime. `Arbiter` is `Clone` and shares its
connection pool; `.http_client()` takes your own `reqwest::Client` if you need
a specific proxy, pool, or set of TLS roots.

**Treat the state as untrusted input.** It is usually something a third party
wrote, and whoever writes it can try to steer the answer. A decision is a
signal, not an authorization: do not let one gate access, move money, or grant
privilege by itself.

## Providers and credentials

The same model is reachable two ways, and they differ in more than a hostname.
A `Provider` is the one place that knows all of it, so you cannot end up with
one provider's endpoint and the other's credential.

| | `Provider::TypeSafe` *(default)* | `Provider::OpenRouter` |
|---|---|---|
| Env var | `TYPESAFE_API_KEY` | `OPENROUTER_API_KEY` |
| Keychain service | `typesafe` | `openrouter` |
| Pinned model | `jev-1.13.0` | `typesafe/jev-1.13` |
| Latest alias | `jev-latest` | `~typesafe/jev-latest` |
| Base-URL override | `TYPESAFE_BASE_URL` | `OPENROUTER_BASE_URL` |
| Extra request fields | none | `session_id`, `user` |

Model slugs are **not** interchangeable between the two, so the default model
follows the provider — prefer `Provider::pinned_model()` over naming one by
hand. Both defaults are pinned versions rather than floating aliases, so a
confidence threshold tuned against the default keeps meaning what it meant.

Routing through OpenRouter is how you reach a decision model other than the
default. `--model` takes any slug the provider serves, and an unknown one is
rejected by the provider rather than by this crate, so a new model needs no
release here:

```sh
arbiter noul "Is this a duplicate charge?" --stdin --open-router \
    --model '~typesafe/jev-latest' < order.json
```

### Where the key comes from

```rust
let arbiter = Arbiter::new(key)?;            // you supply it; nothing ambient
let arbiter = Arbiter::discover(provider)?;  // environment, then Keychain
let arbiter = Arbiter::from_env(provider)?;  // environment only
```

`Arbiter::new` reads nothing from the environment: an embedded tool must not
pick up ambient configuration behind its host's back. `discover` checks the
provider's environment variable, treating a blank value as unset so an
accidentally exported empty variable does not shadow a working key, then on
macOS the login Keychain. Implement `SecretSource` to add a store this crate
has never heard of, and pass it with `Arbiter::with_secrets`.

```sh
security add-generic-password -s typesafe   -a "$USER" -w
security add-generic-password -s openrouter -a "$USER" -w
```

`-w` with no value makes `security` prompt, so the key stays out of your shell
history and out of the process table.

What is guarded, all of it covered by tests:

- A lookup distinguishes **absent** from **broken**. Exit 44 from `security`
  (item not found) falls through to the next source; anything else — notably a
  locked keychain — is an error, so a locked keychain never looks like a
  missing key.
- The key returns on the child's **stdout**, never argv, so it stays out of the
  process table. A test asserts it never appears in a request body either.
- `security` is invoked by **absolute path**, so a hostile `PATH` cannot
  impersonate it.
- The key is held in `Secret`, whose `Debug`, `Display` and serialized forms
  are `[redacted]`. It is **not** scrubbed from memory on drop — redaction
  prevents accidental logging, not heap inspection.

## CLI

```sh
cargo build --features cli
```

```sh
arbiter choice "Which team should own this ticket?" \
    --option 'payments=Checkout and billing issues.' \
    --option 'frontend=Rendering and browser issues.' \
    --state-file ticket.json \
    --min-confidence 0.75

arbiter noul "Is this a duplicate charge?" --stdin < order.json

arbiter score "How urgent is this?" \
    --level "Can wait" --level "Today" --level "Within the hour" \
    --state "Customer cannot check out"

arbiter batch request.json --json
```

State comes from `--state`, `--state-file`, or `--stdin`. The first three
subcommands ask exactly one question; `batch` reads a complete request with
`state` and `questions` and fills in `model` when the file omits it, which is
how you ask several questions about one state in a single round trip. Global
flags: `--open-router`, `--model`, `--env-only`.

Exit codes: `0` answered, `3` answered but under the requested threshold, `1`
failed, `2` usage error. The `3` is the useful one in a pipeline — it separates
"the model was unsure, ask a human" from "the call broke".

## Example — `examples/cti_mcq.rs` and `examples/choice_report.rs`

One worked example, pointed at [CTI-Bench]'s `cti-mcq` split: 2,500
multiple-choice questions on cyber threat intelligence. Each becomes one
`choice` question whose options are the four candidate answers, keyed by
letter — `A`/`B`/`C`/`D`, never `1`/`2`/`3`/`4`, because a numeric key can
collide with the content of an answer.

Asking and drawing the page are two executables, because each talks to
something different:

| Executable | Mode | Reads | Writes | Network |
|---|---|---|---|---|
| `cti_mcq` | `fetch` | — | `cti-mcq.tsv` | Hugging Face |
| `cti_mcq` | `run` | `cti-mcq.tsv` | `cti-mcq-run.json` | the provider |
| `choice_report` | — | `cti-mcq-run.json` | `cti-mcq-report.html` | none |

`fetch` and `run` default to reading what the one before it wrote, so the
normal path names no paths; `--in` and `--out` override either end. All three
files land in the working directory and are gitignored.

```sh
make datasets                                               # fetch the split
make cti-mcq-run                                            # ask, and record
make cti-mcq-render                                         # draw the page, free

cargo run --release --example cti_mcq -- run --all          # every question
cargo run --release --example cti_mcq -- run --attempts 1   # single shot, cheap
cargo run --release --example cti_mcq -- run --open-router --model '~typesafe/jev-latest'
cargo run --release --example choice_report -- --limit 50   # just the first 50
```

Splitting `run` from `render` is the point: inference costs money and rendering
does not, so the page can be reworked as many times as it takes without paying
again, and last week's run can be redrawn by today's renderer. The record is the
general `choice-eval` schema in [`arbiter::eval`](src/eval.rs) — not a
CTI-specific one — so any harness that asks a choice question can write one and
the same renderer can draw it. It holds every attempt and both halves of every
exchange, carries a `schema` field, and `choice_report` refuses a version it
does not know.

### One question per request, on purpose

This example does **not** batch, even though the API is at its cheapest when it
does, because batching means questions sharing one state and these questions
share nothing. Each carries its own prompt and its own four options. Putting
twenty of them in one request would show every question the other nineteen —
which both inflates the tokens each judgement is billed for and lets one
question's options bleed into another's answer, so the accuracy would no longer
be comparable to the benchmark's own numbers. Concurrency, not batching, is
what keeps the run fast: `--concurrency` requests are in flight at once.

Batching would be right the moment you asked *several* things about one CTI
question — whether it is ambiguous, which ATT&CK tactic it concerns, how it
should be answered — because those share a state. That is what `arbiter batch`
and a multi-question `Request` are for.

### Fetching the split

`fetch` is one GET of a static TSV — no dataset library and no Python.
`hf://datasets/AI4Sec/cti-bench/cti-mcq.tsv`, which is how the Python ecosystem
names it, is sugar for
`https://huggingface.co/datasets/AI4Sec/cti-bench/resolve/main/cti-mcq.tsv`.
The file quotes nothing, embeds no newlines and carries exactly eight
tab-separated fields on every line, so reading it needs no CSV parser either;
columns are resolved by header name and a ragged line is skipped rather than
guessed at. Five rows are missing an option and are skipped — `fetch` says how
many.

### pass^k

`run` asks each question five times by default, because one answer cannot tell
you whether a decision is *reliable*. The report turns that into `pass^k`: the
share of questions answered correctly on **all** of k attempts.

Note the caret. `pass@k` asks whether *any* of k attempts succeeds and rewards
a lucky guess; `pass^k` asks whether *every* one does, which is the question
worth asking of a decision you intend to act on unsupervised. The gap between
`pass^1` and `pass^5` is how much of the accuracy was luck. The estimator is
the hypergeometric one — `C(c, k) / C(n, k)` per question, averaged — not
`(c/n)^k`, which is biased.

Five attempts is five times the requests. `--attempts 1` is the cheap
single-shot run, and the plan line prints the request count before any of it is
spent.

### The page

One self-contained HTML file: no assets, light and dark. It leads with the
count and `pass^k`, then gives one block per question:

- the question, and the four options with how often each was chosen and the
  mean probability the model gave it
- one row per attempt: the letter, the confidence, the margin, the latency
- **the payload posted and the response received**, verbatim, one click away

Filters narrow it to the questions that got an attempt wrong, the ones whose
attempts disagreed with each other, or the unconfident ones — usually where the
interesting reading is. Expect roughly 5.5 KB of record and 6.5 KB of page per
question at five attempts, so the full 2,500 is about 14 MB and 17 MB;
`choice_report --limit N` keeps the page small while the record stays complete.

Sampling is seeded and uniform rather than a leading slice — the split is
ordered by source technique, so a contiguous block would ask about one corner
of ATT&CK and report it as accuracy. A rerun asks the same questions; pass a
different `--seed` for an independent sample.

[CTI-Bench]: https://arxiv.org/abs/2406.07599

## Development

```sh
make            # list targets
make test       # unit, integration and doc tests
make lint       # formatting, clippy as errors, and the no-features build
make pre-commit # format, then lint, test and doc — the gate before committing
```

The suite runs **entirely offline** and needs no API key. The arbiter is driven
end to end against a stub endpoint built from a plain `TcpListener`, so the real
request is serialised, sent and parsed — including auth headers, every error
status, and the malformed-response path. Both providers are covered, and the
request and response fixtures are copied verbatim from the published API
references, so a drift in a documented shape shows up as a test failure. The
Keychain source is covered on any Unix host by pointing it at a stub script.

`make lint` includes `check-embed`, which compiles the library with no features
and fails if it ever starts needing a runtime or an argument parser.

Credentials come from the environment or the Keychain — there is no dotenv file
to copy. Binaries land in `$CARGO_TARGET_DIR`, not `./target`.

## Licence

MIT — see [LICENSE](LICENSE).
