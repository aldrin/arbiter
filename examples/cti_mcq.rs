//! Two modes over one benchmark: fetch the split, then ask it.
//!
//! The benchmark is [CTI-Bench]'s `cti-mcq` split — 2,500 multiple-choice
//! questions on cyber threat intelligence: MITRE ATT&CK techniques,
//! mitigations, CVE scoring, and the surrounding vocabulary. Each question
//! becomes one `choice` question whose options are the four candidate answers.
//!
//! ```text
//! cargo run --release --example cti_mcq -- fetch            # get the split
//! cargo run --release --example cti_mcq -- run              # ask, and record
//! cargo run --release --example choice_report               # draw the page
//! ```
//!
//! `fetch` is one GET of a static TSV on Hugging Face — no dataset library and
//! no Python. `run` reads the local file, draws a seeded sample, asks each
//! question `--attempts` times, and writes one JSON record. Drawing the page
//! from that record is a separate executable (`choice_report`), because
//! inference costs money and rendering does not: the page can be reworked as
//! many times as it takes without paying again.
//!
//! The record is the general `choice-eval` schema in [`arbiter::eval`], not a
//! CTI-specific one, so any harness that asks a choice question can write one
//! and the same renderer can draw it.
//!
//! # Asking more than once
//!
//! `run` asks each question `--attempts` times, five by default, because one
//! answer cannot tell you whether a decision is *reliable*. The report turns
//! that into `pass^k`: the share of questions answered correctly on **all** of
//! k attempts. Note the caret — `pass@k` asks whether *any* attempt succeeds
//! and rewards a lucky guess, while `pass^k` asks whether every one does,
//! which is the question worth asking of a decision you intend to act on
//! unsupervised. The gap between `pass^1` and `pass^5` is how much of the
//! accuracy was luck.
//!
//! Five attempts is five times the requests. `--attempts 1` is the cheap
//! single-shot run; the plan line prints the request count before any of it is
//! spent.
//!
//! # Options are keyed by letter
//!
//! `A`/`B`/`C`/`D`, never `1`/`2`/`3`/`4` and never the answer text itself. A
//! numeric key can collide with the content of an answer — the CTI-Bench paper
//! hit a numeric key `36` against "which element has atomic number 36" — and
//! keying by the answer text would make the label carry the judgement. A
//! letter is a pure handle.
//!
//! [CTI-Bench]: https://arxiv.org/abs/2406.07599

// Counting turns integers into ratios all over this example. Every count is a
// sample size, far below where an f64 loses integers.
#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arbiter::eval::{Attempt, Question, Run, SCHEMA, Stats, Usage, wilson};
use arbiter::{Arbiter, Provider, Request, SINGLE_QUESTION_NAME as ANSWER};
use serde_json::{Value, json};
use tokio::sync::Semaphore;

/// The benchmark this example is about.
const BENCHMARK: &str = "cti_mcq";

/// Where `fetch` writes the split, and `run` looks for one.
///
/// The working directory, not a subdirectory: it is one file, and every mode
/// reads what the previous one wrote from the same place.
const DEFAULT_SPLIT: &str = "cti-mcq.tsv";

/// Where `run` writes its record, and `choice_report` looks for one.
const DEFAULT_RECORD: &str = "cti-mcq-run.json";

/// The option keys. Letters, not numerals — see the module docs.
const LETTERS: [&str; 4] = ["A", "B", "C", "D"];

/// Published price per million input tokens, in USD. Output is free.
///
/// For estimating what a run cost when the provider reported none of its own —
/// TypeSafe does not, OpenRouter does. A published price drifts, so
/// `usage.cost` wins whenever the provider sent one.
const USD_PER_MILLION_INPUT_TOKENS: f64 = 0.042;

#[tokio::main]
async fn main() -> ExitCode {
    match dispatch().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("cti-mcq: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn dispatch() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1).peekable();
    let mode = args.next().unwrap_or_default();

    match mode.as_str() {
        "fetch" => fetch(&Options::parse(args, Mode::Fetch)?).await,
        "run" => run(&Options::parse(args, Mode::Run)?).await,
        "-h" | "--help" | "help" => {
            usage();
            Ok(())
        }
        "" => {
            usage();
            Err("name a mode: fetch or run".into())
        }
        other => {
            usage();
            Err(format!("unknown mode {other:?}; expected fetch or run").into())
        }
    }
}

fn usage() {
    println!(
        "usage: cti_mcq fetch [--out FILE] [--force]\n\
         \x20      cti_mcq run   [--n COUNT | --all] [--attempts N] [--seed N]\n\
         \x20                    [--concurrency N] [--in FILE] [--out FILE]\n\
         \x20                    [--provider NAME] [--open-router] [--model SLUG]\n\
         \x20                    [--base-url URL]\n\
         \n\
         fetch  downloads the split: one GET of a static TSV.\n\
         run    asks the questions and records every attempt to JSON.\n\
         \n\
         Drawing the page is a separate executable:\n\
         \x20 cargo run --release --example choice_report\n\
         \n\
         Each mode defaults to reading what the one before it wrote:\n\
         \x20 fetch --out {DEFAULT_SPLIT}\n\
         \x20 run   --in {DEFAULT_SPLIT} --out {DEFAULT_RECORD}\n\
         \n\
         defaults: --attempts 5, --n 200, --concurrency 16, --seed 1"
    );
}

/// Which mode's defaults to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Fetch,
    Run,
}

/// Everything either mode was told to do.
///
/// One struct for both, so a flag means the same thing wherever it appears;
/// [`Options::parse`] rejects a flag the chosen mode cannot honour rather
/// than ignoring it.
struct Options {
    /// Questions to draw, unless [`Self::all`].
    n: usize,
    /// Ask every scorable question in the split.
    all: bool,
    /// How many times to ask each question.
    attempts: usize,
    /// The sampling seed.
    seed: u64,
    /// Requests in flight at once.
    concurrency: usize,
    /// `run`'s input split.
    input: String,
    /// Where this mode writes.
    out: String,
    /// Fetch the split again even when it is already on disk.
    force: bool,
    /// Who serves the requests.
    provider: Provider,
    /// A model slug instead of the provider's pinned default.
    model: Option<String>,
    /// A gateway, a staging endpoint, or a stub.
    base_url: Option<String>,
}

impl Options {
    fn parse(
        args: impl Iterator<Item = String>,
        mode: Mode,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut options = Self {
            n: 200,
            all: false,
            // Five, so `pass^5` means something without a flag. One answer
            // cannot say whether a decision is reliable.
            attempts: 5,
            seed: 1,
            concurrency: 16,
            // Each mode reads what the one before it wrote, so the
            // defaults chain and a normal run names no paths at all.
            input: match mode {
                Mode::Fetch => String::new(),
                Mode::Run => DEFAULT_SPLIT.to_owned(),
            },
            out: match mode {
                Mode::Fetch => DEFAULT_SPLIT.to_owned(),
                Mode::Run => DEFAULT_RECORD.to_owned(),
            },
            force: false,
            provider: Provider::default(),
            model: None,
            base_url: None,
        };

        let mut args = args.peekable();
        while let Some(argument) = args.next() {
            let mut value = |flag: &str| -> Result<String, String> {
                args.next().ok_or_else(|| format!("{flag} needs a value"))
            };
            match argument.as_str() {
                "--all" => options.all = true,
                "--n" => {
                    options.n = value("--n")?.parse()?;
                    if options.n == 0 {
                        return Err("--n must be at least 1".into());
                    }
                }
                "--attempts" => {
                    options.attempts = value("--attempts")?.parse()?;
                    if options.attempts == 0 {
                        return Err("--attempts must be at least 1".into());
                    }
                }
                "--seed" => options.seed = value("--seed")?.parse()?,
                "--concurrency" => {
                    options.concurrency = value("--concurrency")?.parse()?;
                    if options.concurrency == 0 {
                        return Err("--concurrency must be at least 1".into());
                    }
                }
                "--force" => options.force = true,
                "--in" => options.input = value("--in")?,
                "--out" => options.out = value("--out")?,
                "--provider" => options.provider = value("--provider")?.parse()?,
                "--open-router" => options.provider = Provider::OpenRouter,
                "--model" => options.model = Some(value("--model")?),
                "--base-url" => options.base_url = Some(value("--base-url")?),
                "-h" | "--help" => {
                    usage();
                    std::process::exit(0);
                }
                other => return Err(format!("unexpected argument {other:?}").into()),
            }
        }

        // Saying something a mode cannot honour is a mistake worth reporting,
        // not something to quietly ignore.
        if mode == Mode::Fetch {
            for (flag, given) in [
                ("--all", options.all),
                ("--open-router", options.provider != Provider::default()),
                ("--model", options.model.is_some()),
                ("--base-url", options.base_url.is_some()),
            ] {
                if given {
                    return Err(format!("{flag} is a run option; fetch downloads the split").into());
                }
            }
        }
        if mode == Mode::Run && options.force {
            return Err("--force is a fetch option; run always asks the questions".into());
        }

        Ok(options)
    }
}

/// A client for the chosen provider, with the host's credential.
fn build_client(options: &Options) -> Result<Arbiter, Box<dyn std::error::Error>> {
    let model = options
        .model
        .clone()
        .unwrap_or_else(|| arbiter::provider::model_from_env_or(options.provider.pinned_model()));
    let mut client = Arbiter::discover(options.provider)?
        .model(model)
        .timeout(REQUEST_TIMEOUT);
    if let Some(base) = &options.base_url {
        client = client.base_url(base.clone());
    }
    Ok(client)
}

/// A counter that rewrites itself in place, only on a terminal.
fn progress(label: &str, done: usize) {
    use std::io::{IsTerminal as _, Write as _};

    let mut out = std::io::stdout();
    if !out.is_terminal() {
        return;
    }
    print!("\r{label} {done}  ");
    let _ = out.flush();
}

// -------------------------------------------------------------------- fetch

// `fetch` mode: get the split, and read it.
//
// The split is a single static TSV on the Hugging Face CDN, so fetching it is
// one GET — no dataset library, no pagination, no rate limit to back off
// from. `hf://datasets/AI4Sec/cti-bench/cti-mcq.tsv`, which is how the
// Python ecosystem names it, is sugar for exactly the URL below.
//
// Kept apart from `run` on purpose: this is the only mode that talks to
// Hugging Face, and a scoring run reads the file from disk and talks to the
// decision API alone. The file lands in the working directory — one file does
// not need a directory of its own — and is gitignored.

/// Where the split lives. A plain file, not an API.
const SOURCE: &str = "https://huggingface.co/datasets/AI4Sec/cti-bench/resolve/main/cti-mcq.tsv";

/// The columns this example needs, by their header name.
///
/// Looked up by name rather than position, so a column added upstream shifts
/// nothing. The split also carries a `Prompt` column holding the dataset's own
/// instruction text; it is deliberately ignored, because the question is posed
/// structurally here and a prose preamble would describe a format this API
/// does not use.
const COLUMNS: [&str; 7] = [
    "URL", "Question", "Option A", "Option B", "Option C", "Option D", "GT",
];

// ------------------------------------------------------------------ the mode

async fn fetch(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    let destination = Path::new(&options.out);

    if destination.is_file() && !options.force {
        let rows = read(&options.out)?;
        println!(
            "{} already holds {} rows; --force to fetch it again",
            destination.display(),
            rows.len()
        );
        return Ok(());
    }

    println!("fetching     {SOURCE}");
    let body = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?
        .get(SOURCE)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;

    // A bare filename has no parent to create; a path given with one does.
    if let Some(parent) = destination.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    // Written to a temporary file and moved into place, so an interrupted
    // fetch cannot leave a half-written split that `run` would then read.
    let partial = destination.with_extension("tsv.part");
    std::fs::write(&partial, &body)?;
    std::fs::rename(&partial, destination)?;

    let rows = parse(&body)?;
    println!(
        "wrote        {} ({} bytes, {} rows)",
        destination.display(),
        body.len(),
        rows.len()
    );
    let scorable = rows.iter().filter(|row| row.is_scorable()).count();
    if scorable < rows.len() {
        println!(
            "note         {} row(s) are missing an option or an answer and \
             will be skipped",
            rows.len() - scorable
        );
    }
    Ok(())
}

// ------------------------------------------------------------------ the data

/// One row of the split.
///
/// Plain strings rather than options: a field the TSV leaves empty arrives as
/// an empty string, and [`Row::is_scorable`] rejects it either way. Five rows
/// of 2,500 are missing an option, and one records its answer in lower case.
#[derive(Debug, Default, Clone)]
struct Row {
    url: String,
    question: String,
    options: [String; 4],
    gold: String,
}

impl Row {
    /// The four candidate answers, in letter order.
    fn options(&self) -> [&str; 4] {
        [
            self.options[0].as_str(),
            self.options[1].as_str(),
            self.options[2].as_str(),
            self.options[3].as_str(),
        ]
    }

    /// The correct letter, or `None` when the row cannot be scored.
    ///
    /// Uppercased because one row records its answer as `b`. Case is not a
    /// judgement, so normalising it scores the row rather than discarding it.
    fn gold_letter(&self) -> Option<&'static str> {
        let gold = self.gold.trim().to_ascii_uppercase();
        LETTERS.iter().copied().find(|letter| *letter == gold)
    }

    /// Whether every field this run needs is present.
    ///
    /// An option with no text cannot be offered: the model would be asked to
    /// choose between a label and nothing.
    fn is_scorable(&self) -> bool {
        !self.question.trim().is_empty()
            && self.options().iter().all(|text| !text.trim().is_empty())
            && self.gold_letter().is_some()
    }

    /// The request that asks this question.
    ///
    /// Identical on every attempt — the variation being measured is the
    /// model's, not the request's.
    fn request(&self) -> Request {
        Request::new(json!({ "question": self.question })).choice(
            ANSWER,
            INSTRUCTIONS,
            LETTERS
                .iter()
                .copied()
                .zip(self.options())
                .map(|(letter, text)| (letter, json!(text))),
        )
    }
}

/// Read the split `fetch` wrote.
fn read(path: &str) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        format!("no split at {path} ({error}); fetch it first with `make datasets`")
    })?;
    parse(&text)
}

/// Rows out of the TSV.
///
/// A deliberate `split('\t')` rather than a CSV parser: this file quotes
/// nothing, embeds no newlines, and carries exactly eight fields on every
/// line, so a quoting-aware reader would only add a dependency and a way to
/// disagree with the file. A line with the wrong field count is skipped rather
/// than guessed at.
fn parse(text: &str) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    let mut lines = text.lines();
    let header = lines.next().ok_or("the split is empty")?;
    let names: Vec<&str> = header.split('\t').map(str::trim).collect();

    // Resolve each column once, by name.
    let mut at = [0_usize; COLUMNS.len()];
    for (slot, wanted) in COLUMNS.iter().enumerate() {
        at[slot] = names
            .iter()
            .position(|name| name == wanted)
            .ok_or_else(|| format!("the split has no {wanted:?} column; found {names:?}"))?;
    }

    let mut rows = Vec::new();
    let mut malformed = 0;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != names.len() {
            malformed += 1;
            continue;
        }
        let field = |slot: usize| fields[at[slot]].to_owned();
        rows.push(Row {
            url: field(0),
            question: field(1),
            options: [field(2), field(3), field(4), field(5)],
            gold: field(6),
        });
    }
    if malformed > 0 {
        eprintln!("  {malformed} line(s) had the wrong field count and were skipped");
    }
    Ok(rows)
}

// ---------------------------------------------------------------------- run

// `run` mode: ask the questions, record everything, spend money.
//
// The only mode that contacts the provider. It reads the exported split,
// draws a seeded sample, asks each question `--attempts` times, and writes
// one JSON record in the general `choice-eval` schema.

/// What the model is asked about each question.
const INSTRUCTIONS: &str = "Which option is the correct answer to `question`?";

// ------------------------------------------------------------------ the mode

async fn run(options: &Options) -> Result<(), Box<dyn std::error::Error>> {
    let all = read(&options.input)?;
    let exported = all.len();
    let scorable: Vec<Row> = all.into_iter().filter(Row::is_scorable).collect();
    if scorable.is_empty() {
        return Err(format!("no scorable rows in {}", options.input).into());
    }
    let unscorable = exported - scorable.len();
    if unscorable > 0 {
        println!("skipped:     {unscorable} of {exported} rows missing an option or an answer");
    }

    let total_available = scorable.len();
    let wanted = if options.all {
        total_available
    } else {
        options.n.min(total_available)
    };
    let rows = sample(scorable, wanted, options.seed);

    let client = Arc::new(build_client(options)?);
    println!("provider:    {}", client.provider());
    println!("model:       {}", client.model_id());
    println!(
        "plan:        {} questions x {} attempts = {} requests, seed {}, {} concurrent",
        rows.len(),
        options.attempts,
        rows.len() * options.attempts,
        options.seed,
        options.concurrency
    );
    println!();

    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let clock = Instant::now();
    let (questions, usage, failures) = ask(&client, rows, options).await;
    let wall = clock.elapsed();

    if questions.is_empty() {
        return Err("every request failed; nothing to record".into());
    }

    let run = Run {
        schema: SCHEMA.to_owned(),
        started_unix_ms: u64::try_from(started.as_millis()).unwrap_or(u64::MAX),
        benchmark: BENCHMARK.to_owned(),
        provider: client.provider().to_string(),
        model: client.model_id().to_owned(),
        seed: options.seed,
        attempts: options.attempts,
        concurrency: options.concurrency,
        asked: questions.len(),
        available: total_available,
        skipped: unscorable,
        failures,
        wall_ms: u64::try_from(wall.as_millis()).unwrap_or(u64::MAX),
        usage,
        questions,
    };

    report(&run);

    // Pretty-printed: the file is meant to be readable and diffable, and a
    // run is written once and read many times.
    std::fs::write(&options.out, serde_json::to_string_pretty(&run)?)?;
    println!("\nrecord:      {}", options.out);
    println!(
        "render it:   cargo run --release --example choice_report -- --in {}",
        options.out
    );
    Ok(())
}

/// What the run found, for a terminal.
fn report(run: &Run) {
    let stats = Stats::of(run);
    let (dollars, reported) = run.usage.dollars(USD_PER_MILLION_INPUT_TOKENS);

    let (low, high) = wilson(stats.correct_attempts, stats.decided_attempts);
    println!(
        "correct:     {}/{} attempts ({:.1}%, 95% CI {:.1}–{:.1}%)",
        stats.correct_attempts,
        stats.decided_attempts,
        stats.accuracy() * 100.0,
        low * 100.0,
        high * 100.0,
    );

    if stats.pass_hat.len() > 1 {
        let line: Vec<String> = stats
            .pass_hat
            .iter()
            .map(|(k, value)| format!("pass^{k} {:.1}%", value * 100.0))
            .collect();
        println!("reliability: {}", line.join(", "));
        println!(
            "agreement:   {}/{} unanimous, {}/{} correct by majority",
            stats.unanimous, stats.questions, stats.majority_correct, stats.questions
        );
    }

    if let Some(mean) = stats.mean_confidence {
        println!("confidence:  {:.1}% mean", mean * 100.0);
    }
    println!(
        "run:         {:.1}s wall, {}ms median latency{}",
        run.wall_ms as f64 / 1000.0,
        stats.median_latency_ms,
        if run.failures > 0 {
            format!(", {} failed", run.failures)
        } else {
            String::new()
        }
    );
    println!(
        "spend:       {} input tokens, ${dollars:.6} {}",
        run.usage.input_tokens,
        if reported { "reported" } else { "estimated" }
    );
}

/// Ask every question `attempts` times, bounded to `concurrency` in flight.
///
/// Every attempt is its own task, so the concurrency limit covers the whole
/// run rather than one question at a time — a run of 200 questions at 5
/// attempts saturates the limit even if some questions are slow.
async fn ask(
    client: &Arc<Arbiter>,
    rows: Vec<Row>,
    options: &Options,
) -> (Vec<Question>, Usage, usize) {
    let permits = Arc::new(Semaphore::new(options.concurrency));
    let mut tasks = tokio::task::JoinSet::new();

    // The question text, options and payload do not vary by attempt, so they
    // are built once here and the tasks only carry what they need.
    let mut shells: Vec<Question> = rows
        .iter()
        .enumerate()
        .map(|(index, row)| Question {
            index,
            text: row.question.clone(),
            source: (!row.url.trim().is_empty()).then(|| row.url.clone()),
            options: LETTERS
                .iter()
                .copied()
                .zip(row.options())
                .map(|(letter, text)| (letter.to_owned(), text.to_owned()))
                .collect(),
            expected: row.gold_letter().unwrap_or("A").to_owned(),
            payload: Value::Null,
            attempts: Vec::with_capacity(options.attempts),
        })
        .collect();

    for (index, row) in rows.into_iter().enumerate() {
        let gold = row.gold_letter().unwrap_or("A").to_owned();
        let request = row.request();
        for ordinal in 1..=options.attempts {
            let client = Arc::clone(client);
            let permits = Arc::clone(&permits);
            let request = request.clone();
            let gold = gold.clone();
            tasks.spawn(async move {
                let _permit = permits.acquire().await;
                let (payload, attempt, spent) = once(&client, request, &gold, ordinal).await;
                (index, payload, attempt, spent)
            });
        }
    }

    let mut usage = Usage::default();
    let mut failures = 0;
    let mut done = 0;
    while let Some(finished) = tasks.join_next().await {
        match finished {
            Ok((index, payload, attempt, spent)) => {
                if let Some(spent) = &spent {
                    usage.record(spent);
                }
                if let Some(message) = &attempt.error {
                    failures += 1;
                    eprintln!("  #{} attempt {}: {message}", index + 1, attempt.ordinal);
                }
                if let Some(shell) = shells.get_mut(index) {
                    // Every attempt posts the same body, so the first one to
                    // land supplies it and the rest agree.
                    if let Some(payload) = payload
                        && shell.payload.is_null()
                    {
                        shell.payload = payload;
                    }
                    shell.attempts.push(attempt);
                }
            }
            Err(error) => {
                failures += 1;
                eprintln!("  task failed: {error}");
            }
        }
        done += 1;
        progress("  asking", done);
    }
    println!();

    // Tasks finish out of order; the record is read by people.
    for shell in &mut shells {
        shell.attempts.sort_by_key(|a| a.ordinal);
    }
    shells.retain(|shell| !shell.attempts.is_empty());
    (shells, usage, failures)
}

/// One ask: the payload it posted, what it decided, and what it cost.
///
/// The payload comes back so the caller can store it once per question rather
/// than once per attempt — every attempt posts the same bytes.
async fn once(
    client: &Arbiter,
    request: Request,
    gold: &str,
    ordinal: usize,
) -> (Option<Value>, Attempt, Option<arbiter::Usage>) {
    let started = Instant::now();
    let recorded = client.ask_recorded(request).await;
    let latency_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);

    let exchange = match recorded {
        Ok(exchange) => exchange,
        Err(error) => {
            return (
                None,
                Attempt::failed(ordinal, latency_ms, error.to_string()),
                None,
            );
        }
    };

    let response = Some(as_json(&exchange.response));
    let usage = Some(exchange.answers.usage);
    match exchange.answers.choice(ANSWER) {
        Ok(choice) => {
            let chosen = choice.choice.trim().to_ascii_uppercase();
            (
                Some(exchange.request),
                Attempt {
                    ordinal,
                    correct: Some(chosen == gold),
                    chosen: Some(chosen),
                    confidence: choice.confidence,
                    margin: choice.margin(),
                    probabilities: choice.probabilities.clone(),
                    latency_ms,
                    response,
                    error: None,
                },
                usage,
            )
        }
        // Answered, but not with the question we asked: worth recording as an
        // attempt with a reason rather than dropping, and worth keeping the
        // response that came back.
        Err(error) => (
            Some(exchange.request),
            Attempt {
                response,
                ..Attempt::failed(ordinal, latency_ms, error.to_string())
            },
            usage,
        ),
    }
}

/// A response body as JSON, or as a JSON string when it is not JSON at all.
///
/// A gateway or an edge can answer with HTML, and the point of recording the
/// response is to record what actually arrived.
fn as_json(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_owned()))
}

// ------------------------------------------------------------------ sampling

/// `n` rows drawn uniformly from `all`, reproducibly.
///
/// Uniformly rather than as a leading block: the split is ordered by source
/// technique, so a contiguous slice would ask about one corner of ATT&CK and
/// report it as the model's accuracy. Seeded, so a rerun asks the same
/// questions and two runs are comparable.
fn sample(mut all: Vec<Row>, n: usize, seed: u64) -> Vec<Row> {
    let total = all.len();
    if n >= total {
        return all;
    }
    let mut indices = distinct_indices(total, n, seed);
    indices.sort_unstable();
    let mut chosen = Vec::with_capacity(n);
    for index in indices.into_iter().rev() {
        chosen.push(all.swap_remove(index));
    }
    chosen.reverse();
    chosen
}

/// `count` distinct indices below `total`, from a seeded generator.
fn distinct_indices(total: usize, count: usize, seed: u64) -> Vec<usize> {
    let mut rng = Rng::new(seed);
    let mut swapped: HashMap<usize, usize> = HashMap::new();
    let mut taken = Vec::with_capacity(count);
    // Partial Fisher-Yates over a sparse map: distinct by construction, and
    // cheap even when `total` dwarfs `count`.
    for step in 0..count {
        let pick = step + rng.below(total - step);
        let value = swapped.get(&pick).copied().unwrap_or(pick);
        let head = swapped.get(&step).copied().unwrap_or(step);
        swapped.insert(pick, head);
        taken.push(value);
    }
    taken
}

/// xorshift64*, so a seed is all the reproducibility this needs.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        usize::try_from(self.next_u64() % bound as u64).unwrap_or(0)
    }
}

/// How long a single request may take.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[cfg(test)]
mod tests {
    use super::*;

    /// The shape the real file has: a header, tab-separated, nothing quoted.
    const SAMPLE: &str = "URL\tQuestion\tOption A\tOption B\tOption C\tOption D\tPrompt\tGT\n\
         https://x/1\tWhich one?\tfirst\tsecond\tthird\tfourth\tignored prose\tC\n\
         https://x/2\tAnd this?\tone\ttwo\tthree\t\tignored prose\tb\n";

    #[test]
    fn reads_the_columns_it_needs_and_ignores_the_rest() {
        let rows = parse(SAMPLE).expect("parses");
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].question, "Which one?");
        assert_eq!(rows[0].options(), ["first", "second", "third", "fourth"]);
        assert_eq!(rows[0].gold_letter(), Some("C"));
        assert_eq!(rows[0].url, "https://x/1");
        assert!(rows[0].is_scorable());
    }

    #[test]
    fn a_lower_case_answer_still_scores() {
        // One row of the real 2,500 records its answer as `b`.
        let rows = parse(SAMPLE).expect("parses");
        assert_eq!(rows[1].gold_letter(), Some("B"));
    }

    #[test]
    fn a_row_missing_an_option_is_not_scorable() {
        // Five rows of the real 2,500 leave an option empty.
        let rows = parse(SAMPLE).expect("parses");
        assert!(!rows[1].is_scorable(), "option D is empty");
    }

    #[test]
    fn columns_are_found_by_name_not_position() {
        let reordered = "GT\tOption D\tOption C\tOption B\tOption A\tQuestion\tURL\n\
                         A\tfourth\tthird\tsecond\tfirst\tWhich one?\thttps://x/1\n";
        let rows = parse(reordered).expect("parses");
        assert_eq!(rows[0].options(), ["first", "second", "third", "fourth"]);
        assert_eq!(rows[0].gold_letter(), Some("A"));
    }

    #[test]
    fn a_missing_column_is_an_error_naming_it() {
        let message = parse("URL\tQuestion\n-\t-\n")
            .expect_err("no option columns")
            .to_string();
        assert!(message.contains("Option A"), "{message}");
    }

    #[test]
    fn a_short_line_is_skipped_rather_than_guessed_at() {
        let truncated = format!("{SAMPLE}https://x/3\tonly two fields\n");
        let rows = parse(&truncated).expect("parses");
        assert_eq!(rows.len(), 2, "the ragged line is dropped");
    }

    #[test]
    fn a_quote_in_the_text_is_just_a_character() {
        // The real file holds 70 double quotes, none of them CSV quoting.
        let quoted = "URL\tQuestion\tOption A\tOption B\tOption C\tOption D\tGT\n\
                      -\tAbuse of \"Setuid\" by whom?\ta\tb\tc\td\tA\n";
        let rows = parse(quoted).expect("parses");
        assert_eq!(rows[0].question, "Abuse of \"Setuid\" by whom?");
    }
}
