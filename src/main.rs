//! Ask for a structured decision from the command line.
//!
//! ```text
//! arbiter choice "Which team owns this ticket?" \
//!     --option 'payments=Checkout, billing, or payment processing' \
//!     --option 'frontend=Rendering, layout, or browser compatibility' \
//!     --state-file ticket.json \
//!     --min-confidence 0.75
//! ```

use std::io::{IsTerminal as _, Read as _};
use std::process::ExitCode;

use arbiter::{
    Answers, Arbiter, Provider, Request, SINGLE_QUESTION_NAME as QUESTION_NAME, SecretChain,
};
use clap::{Args, Parser, Subcommand};
use serde_json::Value;

/// Ask a decision model a typed question about some state.
#[derive(Debug, Parser)]
#[command(name = "arbiter", version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    #[command(flatten)]
    creds: Creds,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Ask which of several options applies.
    Choice {
        /// The question to ask.
        question: String,

        /// An option, as `label` or `label=description`. Repeat; at least two.
        #[arg(short, long = "option", value_name = "LABEL[=DESC]", required = true)]
        options: Vec<String>,

        #[command(flatten)]
        common: Common,

        /// Exit 3 when confidence falls below this.
        #[arg(long, value_name = "0.0-1.0")]
        min_confidence: Option<f64>,
    },

    /// Ask a yes/no question.
    Noul {
        /// The question to ask.
        question: String,

        /// What makes the answer true.
        #[arg(long, value_name = "TEXT", requires = "when_false")]
        when_true: Option<String>,

        /// What makes the answer false.
        #[arg(long, value_name = "TEXT", requires = "when_true")]
        when_false: Option<String>,

        #[command(flatten)]
        common: Common,

        /// Exit 3 unless the answer is yes at or above this probability.
        #[arg(long, value_name = "0.0-1.0")]
        min_yes: Option<f64>,
    },

    /// Ask where on an ordered scale the state falls.
    Score {
        /// The question to ask.
        question: String,

        /// A level, lowest first. Repeat; at least two.
        #[arg(short, long = "level", value_name = "TEXT", required = true)]
        levels: Vec<String>,

        #[command(flatten)]
        common: Common,

        /// Exit 3 when confidence falls below this.
        #[arg(long, value_name = "0.0-1.0")]
        min_confidence: Option<f64>,
    },

    /// Post a complete decision request body, for a batch of questions.
    ///
    /// The file must contain `state` and `questions`; `model` is filled in
    /// when absent. This is the only way to ask several questions at once,
    /// which is how the API is meant to be used.
    Batch {
        /// A JSON file holding the request body, or `-` for stdin.
        #[arg(value_name = "FILE")]
        file: String,

        #[command(flatten)]
        common: Common,
    },
}

/// How to reach the API. Shared by every subcommand.
#[derive(Debug, Args)]
struct Creds {
    /// Go through OpenRouter instead of TypeSafe directly.
    ///
    /// Switches the endpoint, the credential, and the model vocabulary
    /// together: the key is read from $OPENROUTER_API_KEY or the `openrouter`
    /// Keychain entry rather than $TYPESAFE_API_KEY or `typesafe`.
    #[arg(long = "open-router", global = true)]
    open_router: bool,

    /// The model to decide with. Defaults to $ARBITER_MODEL, else the
    /// provider's pinned default. Slugs differ between providers.
    #[arg(short, long, global = true)]
    model: Option<String>,

    /// Take the API key only from the environment, never the Keychain.
    #[arg(long, global = true)]
    env_only: bool,
}

impl Creds {
    /// Which provider these options select.
    fn provider(&self) -> Provider {
        if self.open_router {
            Provider::OpenRouter
        } else {
            Provider::TypeSafe
        }
    }
}

/// Options the question subcommands accept.
#[derive(Debug, Args)]
struct Common {
    /// The state to evaluate, as a plain string.
    #[arg(short, long, value_name = "TEXT", conflicts_with_all = ["state_file", "stdin"])]
    state: Option<String>,

    /// A JSON file holding the state.
    #[arg(long, value_name = "FILE", conflicts_with_all = ["state", "stdin"])]
    state_file: Option<String>,

    /// Read the state from standard input.
    #[arg(long, conflicts_with_all = ["state", "state_file"])]
    stdin: bool,

    /// Print the raw response as JSON.
    #[arg(long)]
    json: bool,
}

/// The exit code for an answer the caller should not act on unreviewed.
///
/// Not 2: clap already exits 2 for a usage error, and a script needs to tell
/// "the model was unsure" apart from "you invoked me wrongly".
const EXIT_LOW_CONFIDENCE: u8 = 3;

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "arbiter=warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            eprintln!("arbiter: could not start the async runtime: {error}");
            return ExitCode::FAILURE;
        }
    };

    match runtime.block_on(run()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("arbiter: {error}");
            let mut source = std::error::Error::source(&*error);
            while let Some(cause) = source {
                eprintln!("  caused by: {cause}");
                source = cause.source();
            }
            ExitCode::FAILURE
        }
    }
}

/// What the answer has to clear for a zero exit.
#[derive(Debug, Clone, Copy)]
enum Gate {
    /// The choice must be confident at this threshold.
    Confidence(f64),
    /// The yes/no answer must be yes at this probability.
    Yes(f64),
    /// Nothing to clear.
    Open,
}

impl Gate {
    /// Whether `answers` clears this gate.
    fn passes(self, answers: &Answers) -> Result<bool, Box<dyn std::error::Error>> {
        Ok(match self {
            Self::Confidence(threshold) => match answers.get(QUESTION_NAME) {
                Some(arbiter::Answer::Score(score)) => score.is_confident(threshold),
                _ => answers.choice(QUESTION_NAME)?.is_confident(threshold),
            },
            Self::Yes(threshold) => answers.noul(QUESTION_NAME)?.is_yes(threshold),
            Self::Open => true,
        })
    }
}

/// What to send: either a validated request, or a body to post verbatim.
enum Plan {
    Ask(Request, Gate),
    Raw(Value),
}

async fn run() -> Result<ExitCode, Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let common = cli.command.common();

    // Read the state and validate the request before building a client, so a
    // malformed invocation reports what is actually wrong instead of
    // complaining about a credential it never needed.
    let plan = plan(&cli.command, common)?;

    let client = client(&cli.creds)?;
    let (answers, verdict) = match plan {
        Plan::Ask(request, gate) => {
            let answers = client.ask(request).await?;
            let passed = gate.passes(&answers)?;
            (answers, passed)
        }
        Plan::Raw(mut body) => {
            // Fill in the model only when the body has not chosen one.
            if let Value::Object(map) = &mut body {
                map.entry("model")
                    .or_insert_with(|| Value::String(client.model_id().to_owned()));
            }
            (client.ask_raw(body).await?, true)
        }
    };

    if common.json {
        println!("{}", serde_json::to_string_pretty(&answers)?);
    } else {
        print_answers(&answers);
    }

    Ok(if verdict {
        ExitCode::SUCCESS
    } else {
        eprintln!("arbiter: the answer did not meet the confidence threshold");
        ExitCode::from(EXIT_LOW_CONFIDENCE)
    })
}

/// Turn the parsed arguments into something sendable, validating as it goes.
fn plan(command: &Command, common: &Common) -> Result<Plan, Box<dyn std::error::Error>> {
    let plan = match command {
        Command::Choice {
            question,
            options,
            min_confidence,
            ..
        } => {
            let parsed: Vec<(String, Value)> = options
                .iter()
                .map(|option| match option.split_once('=') {
                    Some((label, description)) => {
                        (label.to_owned(), Value::String(description.to_owned()))
                    }
                    None => (option.clone(), Value::Null),
                })
                .collect();

            Plan::Ask(
                Request::new(read_state(common)?).choice(QUESTION_NAME, question.as_str(), parsed),
                min_confidence.map_or(Gate::Open, Gate::Confidence),
            )
        }

        Command::Noul {
            question,
            when_true,
            when_false,
            min_yes,
            ..
        } => {
            let request = Request::new(read_state(common)?).noul_maybe_described(
                QUESTION_NAME,
                question.as_str(),
                when_true.as_deref(),
                when_false.as_deref(),
            );
            Plan::Ask(request, min_yes.map_or(Gate::Open, Gate::Yes))
        }

        Command::Score {
            question,
            levels,
            min_confidence,
            ..
        } => Plan::Ask(
            Request::new(read_state(common)?).score(
                QUESTION_NAME,
                question.as_str(),
                levels.clone(),
            ),
            min_confidence.map_or(Gate::Open, Gate::Confidence),
        ),

        Command::Batch { file, .. } => Plan::Raw(serde_json::from_str(&read_file(file)?)?),
    };

    if let Plan::Ask(request, _) = &plan {
        request.validate()?;
    }
    Ok(plan)
}

impl Command {
    /// The state options, which every subcommand takes.
    fn common(&self) -> &Common {
        match self {
            Self::Choice { common, .. }
            | Self::Noul { common, .. }
            | Self::Score { common, .. }
            | Self::Batch { common, .. } => common,
        }
    }
}

/// Build a client from the credential options.
fn client(creds: &Creds) -> Result<Arbiter, Box<dyn std::error::Error>> {
    let provider = creds.provider();
    let chain = if creds.env_only {
        SecretChain::env_only()
    } else {
        SecretChain::system()
    };

    // The binary, not the library, is what may honour ambient configuration:
    // an explicit flag wins, then $ARBITER_MODEL, then the provider's pinned
    // default.
    let model = creds
        .model
        .clone()
        .unwrap_or_else(|| arbiter::provider::model_from_env_or(provider.pinned_model()));

    let mut client = Arbiter::with_secrets(provider, &chain)?.model(model);

    // The vendor's own SDKs read a base-URL override; honour the same one so
    // a gateway or a staging endpoint needs no code change.
    if let Some(base) = std::env::var(provider.base_url_env())
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
    {
        // The variable's name, not its value: a base URL can carry a
        // credential in its userinfo, and naming the source is what makes an
        // unexpected endpoint diagnosable anyway.
        tracing::debug!(
            variable = provider.base_url_env(),
            "overriding the base URL from the environment"
        );
        client = client.base_url(base);
    }

    Ok(client)
}

/// The state to evaluate.
///
/// A JSON file or stdin is parsed as JSON when it can be, so nested state
/// keeps its structure and questions can reference fields by path. Text that
/// is not JSON is sent as a plain string, which the API also accepts.
fn read_state(common: &Common) -> Result<Value, Box<dyn std::error::Error>> {
    if let Some(state) = &common.state {
        return Ok(Value::String(state.clone()));
    }
    let raw = if let Some(path) = &common.state_file {
        read_file(path)?
    } else if common.stdin {
        read_stdin()?
    } else {
        return Err("no state given: pass --state, --state-file, or --stdin".into());
    };

    // Only an object, array or string counts as structure. A bare scalar is
    // prose that happens to parse: a file holding `null`, `42` or `true` is
    // text to judge, and sending it as JSON null would judge nothing.
    match serde_json::from_str::<Value>(&raw) {
        Ok(value @ (Value::Object(_) | Value::Array(_) | Value::String(_))) => Ok(value),
        _ => Ok(Value::String(raw)),
    }
}

fn read_file(path: &str) -> std::io::Result<String> {
    if path == "-" {
        return read_stdin();
    }
    std::fs::read_to_string(path)
}

fn read_stdin() -> std::io::Result<String> {
    let mut stdin = std::io::stdin();
    if stdin.is_terminal() {
        eprintln!("arbiter: reading from stdin until EOF (ctrl-d)");
    }
    let mut buffer = String::new();
    stdin.read_to_string(&mut buffer)?;
    Ok(buffer)
}

/// Finish a line with the confidence that qualifies it, when there is one.
fn print_confidence(confidence: Option<f64>) {
    match confidence {
        Some(confidence) => println!("  (confidence {confidence:.2})"),
        None => println!(),
    }
}

fn print_answers(answers: &Answers) {
    for (name, answer) in &answers.answers {
        match answer {
            arbiter::Answer::Noul(noul) => {
                println!("{name}: {:.3}", noul.probability);
            }
            arbiter::Answer::Choice(choice) => {
                print!("{name}: {}", choice.choice);
                print_confidence(choice.confidence);
                for (option, probability) in &choice.probabilities {
                    let marker = if *option == choice.choice { '*' } else { ' ' };
                    println!("  {marker} {option:<24} {probability:.3}");
                }
            }
            arbiter::Answer::Score(score) => {
                print!("{name}: {:.2}", score.score);
                print_confidence(score.confidence);
                if let Some(label) = score.nearest_label() {
                    let text = label
                        .as_str()
                        .map_or_else(|| label.to_string(), ToOwned::to_owned);
                    println!("  nearest: {} — {text}", score.nearest_level());
                }
            }
        }
    }

    match answers.usage.cost {
        Some(cost) => eprintln!(
            "\n{} · {} in / {} out tokens · ${cost:.8}",
            answers.model, answers.usage.input_tokens, answers.usage.output_tokens
        ),
        None => eprintln!(
            "\n{} · {} in / {} out tokens",
            answers.model, answers.usage.input_tokens, answers.usage.output_tokens
        ),
    }
}
