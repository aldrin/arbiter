//! Draw a recorded choice-evaluation run as a standalone HTML page.
//!
//! The record is any `arbiter.choice-eval/1` file — what `cti_mcq run` writes,
//! or any harness that asks a choice question. This reads it, contacts nothing,
//! and writes one self-contained page: the count and `pass^k`, then one block
//! per question with every attempt, the options, and both halves of each
//! exchange.
//!
//! It is a separate executable from the harness on purpose. Inference costs
//! money and rendering does not, so the page can be reworked as many times as
//! it takes without paying for inference again, and last week's run can be
//! redrawn by today's renderer.
//!
//! ```text
//! cargo run --release --example choice_report                    # defaults
//! cargo run --release --example choice_report -- --in run.json --out report.html
//! cargo run --release --example choice_report -- --limit 50
//! ```

// Counting turns integers into ratios all over this file. Every count is a
// sample size, far below where an f64 loses integers.
#![allow(clippy::cast_precision_loss)]

use std::collections::BTreeMap;
use std::env;
use std::fmt::Write as _;
use std::process::ExitCode;

use arbiter::eval::{Question, Run, SCHEMA, Stats, format_unix_ms, wilson};

/// Where the renderer looks for a record when `--in` is not given.
///
/// The working directory, matching where the `cti_mcq` harness writes its run.
const DEFAULT_RECORD: &str = "cti-mcq-run.json";

/// Where it writes the page when `--out` is not given.
const DEFAULT_REPORT: &str = "cti-mcq-report.html";

/// Published price per million input tokens, in USD, used to estimate what a
/// run cost when the provider reported none.
///
/// A published price drifts and is model-specific, so it is a flag rather than
/// a fact this tool bakes in; `usage.cost` wins whenever the provider sent one.
const DEFAULT_USD_PER_MILLION_INPUT_TOKENS: f64 = 0.042;

/// Below this mean confidence, the filter calls a question unconfident.
///
/// Presentation only: a filter has to cut somewhere, and nothing is scored
/// against this.
const UNCONFIDENT_BELOW: f64 = 0.70;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("choice-report: {error}");
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    println!(
        "usage: choice_report [--in FILE] [--out FILE] [--limit N]\n\
         \x20                   [--usd-per-million USD]\n\
         \n\
         Draws a standalone HTML page from a recorded choice-eval run: the\n\
         count and pass^k, then one block per question with every attempt and\n\
         both halves of each exchange. Reads only the record; contacts nothing.\n\
         \n\
         defaults: --in {DEFAULT_RECORD}, --out {DEFAULT_REPORT}, no limit,\n\
         \x20         --usd-per-million {DEFAULT_USD_PER_MILLION_INPUT_TOKENS}"
    );
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = DEFAULT_RECORD.to_owned();
    let mut output = DEFAULT_REPORT.to_owned();
    let mut limit = None;
    let mut usd_per_million_input_tokens = DEFAULT_USD_PER_MILLION_INPUT_TOKENS;

    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        let mut value = |flag: &str| -> Result<String, String> {
            args.next().ok_or_else(|| format!("{flag} needs a value"))
        };
        match argument.as_str() {
            "--in" => input = value("--in")?,
            "--out" => output = value("--out")?,
            "--limit" => {
                let parsed: usize = value("--limit")?.parse()?;
                if parsed == 0 {
                    return Err("--limit must be at least 1".into());
                }
                limit = Some(parsed);
            }
            "--usd-per-million" => {
                usd_per_million_input_tokens = value("--usd-per-million")?.parse()?;
            }
            "-h" | "--help" => {
                usage();
                return Ok(());
            }
            other => return Err(format!("unexpected argument {other:?}").into()),
        }
    }

    render(&input, &output, limit, usd_per_million_input_tokens)
}

/// The schema this renderer understands.
fn check_schema(run: &Run) -> Result<(), Box<dyn std::error::Error>> {
    if run.schema == SCHEMA {
        return Ok(());
    }
    Err(format!(
        "record is {:?} but this build reads {:?}; re-run the questions or \
         render with the matching build",
        run.schema, SCHEMA
    )
    .into())
}

fn render(
    input: &str,
    output: &str,
    limit: Option<usize>,
    usd_per_million_input_tokens: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(input)
        .map_err(|error| format!("cannot read {input} ({error}); run the questions first"))?;
    let run: Run = serde_json::from_str(&text)
        .map_err(|error| format!("{input} is not a run record: {error}"))?;
    check_schema(&run)?;

    let stats = Stats::of(&run);
    let html = page(&run, &stats, limit, usd_per_million_input_tokens);
    std::fs::write(output, html)?;

    println!(
        "rendered {} question(s) x {} attempt(s) from {input}",
        run.questions.len(),
        run.attempts
    );
    if let Some(limit) = limit
        && limit < run.questions.len()
    {
        println!("showed the first {limit}; drop --limit for all of them");
    }
    println!("report:   {output}");
    Ok(())
}

// ------------------------------------------------------------------ the page

fn page(
    run: &Run,
    stats: &Stats,
    limit: Option<usize>,
    usd_per_million_input_tokens: f64,
) -> String {
    let shown = limit
        .unwrap_or(run.questions.len())
        .min(run.questions.len());
    // Roughly 6 KB of markup per question at five attempts.
    let mut out = String::with_capacity(8192 + shown * 6144);
    out.push_str(HEAD);
    header(&mut out, run);
    headline(&mut out, run, stats);
    reliability(&mut out, stats);
    transcript(&mut out, run, stats, shown);
    footer(&mut out, run, usd_per_million_input_tokens);
    out.push_str(TAIL);
    out
}

fn header(out: &mut String, run: &Run) {
    out.push_str("<header class=\"page-head\">\n<div class=\"head-row\">\n<div>\n");
    let _ = writeln!(out, "<h1>{}</h1>", esc(&run.benchmark));
    let _ = writeln!(
        out,
        "<p class=\"sub\">Every decision over {} questions from \
         <code>{}</code>, each asked {} time{}, with the exchange that \
         produced it.</p>",
        run.questions.len(),
        esc(&run.benchmark),
        run.attempts,
        if run.attempts == 1 { "" } else { "s" }
    );
    out.push_str("</div>\n");
    out.push_str(
        "<button type=\"button\" id=\"theme\" class=\"ghost\" \
         aria-label=\"Switch between light and dark\">Theme</button>\n",
    );
    out.push_str("</div>\n<dl class=\"meta\">\n");
    for (label, value) in [
        ("Model", esc(&run.model)),
        ("Provider", esc(&run.provider)),
        ("Sampled", format!("{} of {}", run.asked, run.available)),
        ("Attempts each", run.attempts.to_string()),
        ("Seed", run.seed.to_string()),
        ("Started", format_unix_ms(run.started_unix_ms)),
    ] {
        let _ = writeln!(out, "<div><dt>{label}</dt><dd>{value}</dd></div>");
    }
    out.push_str("</dl>\n</header>\n");
}

/// The count, and the few run facts that qualify it.
fn headline(out: &mut String, run: &Run, stats: &Stats) {
    let (low, high) = wilson(stats.correct_attempts, stats.decided_attempts);

    out.push_str("<section class=\"card hero-card\">\n<div class=\"hero\">\n");
    let _ = writeln!(
        out,
        "<p class=\"hero-label\">Correct answers</p>\n\
         <p class=\"hero-value\">{}<span class=\"unit\">/{}</span></p>\n\
         <p class=\"hero-note\">{:.1}% of attempts · 95% CI {:.1}–{:.1}%</p>",
        stats.correct_attempts,
        stats.decided_attempts,
        stats.accuracy() * 100.0,
        low * 100.0,
        high * 100.0,
    );
    out.push_str("</div>\n<div class=\"tiles\">\n");

    if let Some((k, value)) = stats.deepest_k()
        && k > 1
    {
        tile(
            out,
            &format!("pass^{k}"),
            &format!("{:.1}%", value * 100.0),
            Some(&format!("right all {k} times")),
        );
    }
    if stats.attempts_each > 1 {
        tile(
            out,
            "Unanimous",
            &format!("{}", stats.unanimous),
            Some(&format!("of {} questions", stats.questions)),
        );
        tile(
            out,
            "Right by majority",
            &format!("{}", stats.majority_correct),
            Some(&format!("of {} questions", stats.questions)),
        );
    }
    if let Some(mean) = stats.mean_confidence {
        tile(
            out,
            "Mean confidence",
            &format!("{:.1}%", mean * 100.0),
            None,
        );
    }
    tile(
        out,
        "Median latency",
        &format!("{}<span class=\"unit\">ms</span>", stats.median_latency_ms),
        None,
    );
    if run.failures > 0 {
        tile(
            out,
            "Failed attempts",
            &run.failures.to_string(),
            Some(&if stats.decided_questions < stats.questions {
                format!(
                    "{} question(s) never decided",
                    stats.questions - stats.decided_questions
                )
            } else {
                "every question still decided".to_owned()
            }),
        );
    }
    out.push_str("</div>\n</section>\n");
}

fn tile(out: &mut String, label: &str, value: &str, note: Option<&str>) {
    out.push_str("<div class=\"tile\">\n");
    let _ = write!(
        out,
        "<p class=\"tile-label\">{label}</p><p class=\"tile-value\">{value}</p>"
    );
    if let Some(note) = note {
        let _ = write!(out, "<p class=\"note\">{note}</p>");
    }
    out.push_str("\n</div>\n");
}

/// `pass^k` for every depth the run supports.
///
/// Only worth a section when the questions were asked more than once; with a
/// single attempt `pass^1` is just the accuracy already in the hero.
fn reliability(out: &mut String, stats: &Stats) {
    if stats.pass_hat.len() < 2 {
        return;
    }
    out.push_str("<section class=\"card\">\n");
    out.push_str("<h2>Reliability</h2>\n");
    out.push_str(
        "<p class=\"sub\">The share of questions answered correctly on \
         <em>all</em> of k attempts — note the caret, not an at-sign. \
         <code>pass@k</code> asks whether any attempt succeeds and rewards a \
         lucky guess; <code>pass^k</code> asks whether every one does, which \
         is the question worth asking of a decision you intend to act on \
         unsupervised. The drop from <code>pass^1</code> is how much of the \
         accuracy was luck.</p>\n",
    );
    out.push_str("<ul class=\"ladder\">\n");
    for (k, value) in &stats.pass_hat {
        let _ = writeln!(
            out,
            "<li><span class=\"rung\">pass^{k}</span>\
             <span class=\"meter\"><span class=\"fill\" style=\"width:{:.1}%\"></span></span>\
             <span class=\"pct\">{:.1}%</span></li>",
            value * 100.0,
            value * 100.0
        );
    }
    out.push_str("</ul>\n");
    let _ = writeln!(
        out,
        "<p class=\"sub small\">{} of {} questions had every attempt agree, \
         {} were right every single time, and {} came out right on a majority \
         vote.</p>",
        stats.unanimous, stats.questions, stats.unanimous_correct, stats.majority_correct
    );
    out.push_str("</section>\n");
}

/// One block per question: the substance of the page.
fn transcript(out: &mut String, run: &Run, stats: &Stats, shown: usize) {
    out.push_str("<section class=\"card\">\n<div class=\"head-row\">\n<div>\n");
    out.push_str("<h2>Decisions</h2>\n");
    out.push_str(
        "<p class=\"sub\">The question, what the model answered on each \
         attempt and how sure it was, its probability for every option, and \
         both halves of the exchange.</p>\n",
    );
    out.push_str("</div>\n</div>\n");

    out.push_str("<div class=\"filters\" role=\"group\" aria-label=\"Filter decisions\">\n");
    let mut filters: Vec<(&str, &str)> = vec![("all", "All")];
    filters.push(("wrong", "Any attempt wrong"));
    if stats.attempts_each > 1 {
        filters.push(("split", "Attempts disagreed"));
    }
    filters.push(("unsure", "Unconfident"));
    for (key, label) in filters {
        let pressed = if key == "all" { "true" } else { "false" };
        let _ = writeln!(
            out,
            "<button type=\"button\" class=\"chip\" data-filter=\"{key}\" \
             aria-pressed=\"{pressed}\">{label}</button>"
        );
    }
    let _ = writeln!(
        out,
        "<span class=\"shown\" id=\"shown\" aria-live=\"polite\">{shown} shown</span>"
    );
    out.push_str("</div>\n");

    if shown < run.questions.len() {
        let _ = writeln!(
            out,
            "<p class=\"sub small\">Showing the first {shown} of {}; the \
             record holds them all.</p>",
            run.questions.len()
        );
    }

    out.push_str("<ol class=\"decisions\">\n");
    for question in run.questions.iter().take(shown) {
        one(out, question);
    }
    out.push_str("</ol>\n");
    out.push_str(
        "<p class=\"sub small\" id=\"empty\" hidden>Nothing matches that \
         filter.</p>\n",
    );
    out.push_str("</section>\n");
}

fn one(out: &mut String, question: &Question) {
    let decided = question.decided_count();
    let correct = question.correct_count();
    let all_right = decided > 0 && correct == decided;
    let any_wrong = correct < decided || decided < question.attempts.len();
    let split = decided > 1 && !question.is_unanimous();
    let unsure = question
        .mean_confidence()
        .is_none_or(|value| value < UNCONFIDENT_BELOW);

    let _ = writeln!(
        out,
        "<li class=\"decision\" data-wrong=\"{any_wrong}\" data-split=\"{split}\" \
         data-unsure=\"{unsure}\">"
    );

    // Line one: how it went across every attempt.
    out.push_str("<div class=\"dec-head\">\n");
    let _ = write!(out, "<span class=\"num\">#{}</span>", question.index + 1);
    let (mark, class) = if all_right {
        ("&#10003;", "good")
    } else if correct == 0 {
        ("&#10007;", "bad")
    } else {
        ("&#8211;", "mixed")
    };
    let _ = write!(
        out,
        "<span class=\"verdict {class}\"><span aria-hidden=\"true\">{mark}</span> \
         {correct}/{decided} correct</span>"
    );
    if let Some((label, count)) = question.majority() {
        let agreed = if question.is_unanimous() {
            "unanimous".to_owned()
        } else {
            format!("{count}/{decided} agreed")
        };
        let _ = write!(
            out,
            "<span class=\"chipv\">chose {} · {agreed}</span>",
            esc(&label)
        );
    }
    let _ = write!(
        out,
        "<span class=\"chipv faint\">answer {}</span>",
        esc(&question.expected)
    );
    if let Some(mean) = question.mean_confidence() {
        let _ = write!(
            out,
            "<span class=\"chipv\">confidence {:.0}%</span>",
            mean * 100.0
        );
    }
    if let Some(source) = &question.source {
        let _ = write!(
            out,
            "<a class=\"src\" href=\"{}\" rel=\"noreferrer noopener\">source</a>",
            esc(source)
        );
    }
    out.push_str("\n</div>\n");

    let _ = writeln!(out, "<p class=\"question\">{}</p>", esc(&question.text));
    options(out, question);
    attempts(out, question);

    // The payload is identical on every attempt, so it is shown once.
    let _ = writeln!(
        out,
        "<details class=\"data\"><summary>Payload sent (identical on every \
         attempt)</summary><pre>{}</pre></details>",
        esc(&pretty(&question.payload))
    );
    out.push_str("</li>\n");
}

/// The options, with how often each was chosen and the mean probability the
/// model gave it.
///
/// Both are about this one question, so they sit beside the text rather than
/// in an aggregate view.
fn options(out: &mut String, question: &Question) {
    let picks = pick_counts(question);
    let means = mean_probabilities(question);
    out.push_str("<ul class=\"options\">\n");
    for (label, text) in &question.options {
        let mut flags = String::new();
        if *label == question.expected {
            flags.push_str(" gold");
        }
        let chosen = picks.get(label.as_str()).copied().unwrap_or(0);
        if chosen > 0 {
            flags.push_str(" chosen");
        }
        let _ = write!(out, "<li class=\"option{flags}\">");
        let _ = write!(out, "<span class=\"key\">{}</span>", esc(label));
        let _ = write!(out, "<span class=\"text\">{}", esc(text));
        if *label == question.expected {
            out.push_str(" <span class=\"tag\">correct answer</span>");
        }
        if chosen > 0 {
            let _ = write!(out, " <span class=\"tag\">chosen {chosen}&times;</span>");
        }
        out.push_str("</span>");
        match means.get(label.as_str()) {
            Some(value) => {
                let _ = write!(
                    out,
                    "<span class=\"meter\"><span class=\"fill\" \
                     style=\"width:{:.1}%\"></span></span>\
                     <span class=\"pct\">{:.0}%</span>",
                    value * 100.0,
                    value * 100.0
                );
            }
            None => out.push_str("<span class=\"meter\"></span><span class=\"pct\">—</span>"),
        }
        out.push_str("</li>\n");
    }
    out.push_str("</ul>\n");
}

/// One row per attempt, so a question that wobbled shows where.
fn attempts(out: &mut String, question: &Question) {
    out.push_str("<ol class=\"attempts\">\n");
    for attempt in &question.attempts {
        out.push_str("<li class=\"attempt\">");
        let _ = write!(out, "<span class=\"n\">#{}</span>", attempt.ordinal);
        match (&attempt.chosen, attempt.correct) {
            (Some(chosen), Some(correct)) => {
                let (mark, class) = if correct {
                    ("&#10003;", "good")
                } else {
                    ("&#10007;", "bad")
                };
                let _ = write!(
                    out,
                    "<span class=\"verdict {class}\"><span aria-hidden=\"true\">{mark}</span></span>\
                     <span class=\"letter\">{}</span>",
                    esc(chosen)
                );
            }
            _ => out.push_str(
                "<span class=\"verdict bad\"><span aria-hidden=\"true\">!</span></span>\
                 <span class=\"letter\">—</span>",
            ),
        }
        match attempt.confidence {
            Some(confidence) => {
                let _ = write!(out, "<span class=\"pct\">{:.0}%</span>", confidence * 100.0);
            }
            None => out.push_str("<span class=\"pct\">—</span>"),
        }
        match attempt.margin {
            Some(margin) => {
                let _ = write!(out, "<span class=\"pct faint\">{margin:+.2}</span>");
            }
            None => out.push_str("<span class=\"pct faint\">—</span>"),
        }
        let _ = write!(
            out,
            "<span class=\"pct faint\">{} ms</span>",
            attempt.latency_ms
        );
        if let Some(error) = &attempt.error {
            let _ = write!(out, "<span class=\"err\">{}</span>", esc(error));
        }
        if let Some(response) = &attempt.response {
            let _ = write!(
                out,
                "<details class=\"data inline\"><summary>response</summary>\
                 <pre>{}</pre></details>",
                esc(&pretty(response))
            );
        }
        out.push_str("</li>\n");
    }
    out.push_str("</ol>\n");
}

/// How many attempts chose each label.
fn pick_counts(question: &Question) -> BTreeMap<&str, usize> {
    let mut counts = BTreeMap::new();
    for attempt in question.decided() {
        if let Some(chosen) = attempt.chosen.as_deref() {
            *counts.entry(chosen).or_insert(0) += 1;
        }
    }
    counts
}

/// The mean probability the model gave each label, across decided attempts.
fn mean_probabilities(question: &Question) -> BTreeMap<&str, f64> {
    let mut sums: BTreeMap<&str, (f64, usize)> = BTreeMap::new();
    for attempt in question.decided() {
        for (label, probability) in &attempt.probabilities {
            let entry = sums.entry(label.as_str()).or_insert((0.0, 0));
            entry.0 += probability;
            entry.1 += 1;
        }
    }
    sums.into_iter()
        .map(|(label, (sum, count))| (label, sum / count.max(1) as f64))
        .collect()
}

fn footer(out: &mut String, run: &Run, usd_per_million_input_tokens: f64) {
    let (dollars, reported) = run.usage.dollars(usd_per_million_input_tokens);
    out.push_str("<footer class=\"card\">\n<dl class=\"meta wide\">\n");
    for (label, value) in [
        ("Requests", run.usage.calls.to_string()),
        ("Concurrency", run.concurrency.to_string()),
        ("Wall clock", format!("{:.1}s", run.wall_ms as f64 / 1000.0)),
        ("Input tokens", thousands(run.usage.input_tokens)),
        ("Output tokens", thousands(run.usage.output_tokens)),
        (
            if reported { "Cost" } else { "Cost (estimated)" },
            format!("${dollars:.6}"),
        ),
        ("Failed attempts", run.failures.to_string()),
        ("Skipped rows", run.skipped.to_string()),
        ("Schema", esc(&run.schema)),
    ] {
        let _ = writeln!(out, "<div><dt>{label}</dt><dd>{value}</dd></div>");
    }
    out.push_str("</dl>\n");
    out.push_str(
        "<p class=\"sub small\">Cost is estimated from the published input \
         price when the provider reports none. The options are presented to \
         the model keyed by a short label. This page was rendered from a \
         recorded run and contacted nothing.</p>\n",
    );
    out.push_str("</footer>\n");
}

// -------------------------------------------------------------- fragments

fn pretty(value: &serde_json::Value) -> String {
    // A response that was not JSON was recorded as a JSON string; show the
    // text rather than a quoted one-liner.
    if let serde_json::Value::String(text) = value {
        return text.clone();
    }
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// HTML-escape text that came from the dataset or the provider.
fn esc(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(character),
        }
    }
    out
}

/// Everything before the first section: tokens, layout, marks.
///
/// The palette is the data-viz reference instance unchanged — categorical slot
/// 1 for the meters, the status pair for a verdict, and the documented chrome
/// and ink. Dark mode is declared under both the OS media query and the theme
/// attribute, so a toggle wins either way.
const HEAD: &str = r##"<!doctype html>
<html lang="en" data-palette="#2a78d6">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Choice evaluation — decisions</title>
<style>
:root {
  color-scheme: light;
  --plane: #f9f9f7;
  --surface-1: #fcfcfb;
  --surface-2: #f0efec;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --border: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6;
  --track: #cde2fb;
  --critical: #d03b3b;
  --warning: #fab219;
  --good-ink: #006300;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --plane: #0d0d0d;
    --surface-1: #1a1a19;
    --surface-2: #232322;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --border: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5;
    --track: #184f95;
    --good-ink: #0ca30c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --plane: #0d0d0d;
  --surface-1: #1a1a19;
  --surface-2: #232322;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #898781;
  --border: rgba(255, 255, 255, 0.10);
  --series-1: #3987e5;
  --track: #184f95;
  --good-ink: #0ca30c;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 32px 20px 64px;
  background: var(--plane);
  color: var(--text-primary);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 20px;
}
.page-head, .card, footer.card { width: 100%; max-width: 920px; }
.page-head { padding: 0 4px; }
.head-row { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; }
h1 { margin: 0; font-size: 26px; letter-spacing: -0.01em; }
h2 { margin: 0 0 4px; font-size: 17px; letter-spacing: -0.005em; }
.sub { margin: 0; color: var(--text-secondary); max-width: 72ch; }
.sub em { font-style: normal; font-weight: 600; color: var(--text-primary); }
.small { font-size: 13px; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
code { font-size: 0.92em; color: var(--text-secondary); }
.ghost, .chip {
  background: transparent;
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text-secondary);
  padding: 6px 12px;
  font: inherit;
  font-size: 13px;
  cursor: pointer;
}
.ghost:hover, .chip:hover { color: var(--text-primary); }
.chip[aria-pressed="true"] {
  background: var(--surface-2);
  color: var(--text-primary);
  border-color: var(--text-muted);
}

.card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px 22px;
}
.meta {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 10px 20px;
  margin: 14px 0 0;
}
.meta.wide { margin: 0 0 12px; }
.meta div { min-width: 0; }
.meta dt { color: var(--text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
.meta dd { margin: 2px 0 0; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }

.hero-card { display: grid; grid-template-columns: minmax(190px, 230px) 1fr; gap: 26px; align-items: center; }
.hero-label { margin: 0; color: var(--text-muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
.hero-value { margin: 2px 0 0; font-size: 54px; line-height: 1; font-weight: 600; letter-spacing: -0.02em; }
.hero-value .unit { font-size: 22px; font-weight: 500; color: var(--text-secondary); }
.hero-note { margin: 8px 0 0; color: var(--text-secondary); font-size: 13px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); gap: 14px; }
.tile { border-left: 2px solid var(--border); padding-left: 12px; }
.tile-label { margin: 0; color: var(--text-muted); font-size: 12px; }
.tile-value { margin: 3px 0 0; font-size: 21px; font-weight: 600; }
.tile-value .unit { font-size: 13px; font-weight: 500; color: var(--text-secondary); }
.note { margin: 2px 0 0; font-size: 12px; color: var(--text-secondary); }

.ladder { list-style: none; margin: 14px 0 0; padding: 0; display: grid; gap: 6px; }
.ladder li { display: grid; grid-template-columns: 70px 1fr 54px; align-items: center; gap: 12px; }
.rung { color: var(--text-secondary); font-size: 13px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }

/* The meter track is a lighter step of the fill's own ramp. */
.meter { height: 6px; border-radius: 3px; background: var(--track); overflow: hidden; }
.meter .fill { display: block; height: 100%; border-radius: 3px; background: var(--series-1); }
.pct { font-size: 12px; color: var(--text-secondary); text-align: right; font-variant-numeric: tabular-nums; }
.pct.faint { color: var(--text-muted); }

.filters { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 14px 0 4px; }
.shown { color: var(--text-muted); font-size: 12px; margin-left: 4px; }

.decisions { list-style: none; margin: 8px 0 0; padding: 0; }
.decision { border-top: 1px solid var(--border); padding: 16px 0 18px; }
.decision[hidden] { display: none; }
.dec-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.num { color: var(--text-muted); font-size: 12px; font-variant-numeric: tabular-nums; }
.verdict { font-size: 12px; font-weight: 600; }
.verdict.good { color: var(--good-ink); }
.verdict.bad { color: var(--critical); }
.verdict.mixed { color: var(--warning); }
.chipv {
  font-size: 12px;
  color: var(--text-secondary);
  background: var(--surface-2);
  border-radius: 6px;
  padding: 2px 7px;
  font-variant-numeric: tabular-nums;
}
.chipv.faint { color: var(--text-muted); }
a.src { color: var(--text-muted); font-size: 12px; margin-left: auto; }

.question { margin: 10px 0; font-size: 15px; }

.options { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }
.option {
  display: grid;
  grid-template-columns: 22px 1fr 110px 40px;
  align-items: center;
  gap: 10px;
  padding: 4px 6px;
  border-radius: 6px;
  font-size: 13px;
  color: var(--text-secondary);
}
.option.gold { background: var(--surface-2); }
.option .key { color: var(--text-muted); font-weight: 600; text-align: center; }
.option.gold .key, .option.chosen .key { color: var(--text-primary); }
.option .text { min-width: 0; }
.option .tag {
  font-size: 11px;
  color: var(--text-muted);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 0 5px;
  margin-left: 4px;
  white-space: nowrap;
}
.option.gold .tag { color: var(--good-ink); }
.option.chosen .tag { color: var(--text-primary); }

.attempts { list-style: none; margin: 10px 0 0; padding: 0; display: grid; gap: 2px; }
.attempt {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  padding: 3px 6px;
  border-radius: 6px;
  background: var(--surface-2);
}
.attempt .n { color: var(--text-muted); font-variant-numeric: tabular-nums; min-width: 24px; }
.attempt .letter { font-weight: 600; color: var(--text-primary); min-width: 14px; }
.attempt .pct { text-align: left; min-width: 40px; }
.attempt .err { color: var(--critical); overflow-wrap: anywhere; }
.attempt details.inline { margin-left: auto; }

details.data summary { cursor: pointer; color: var(--text-secondary); font-size: 12px; }
details.data summary:hover { color: var(--text-primary); }
details.data { margin-top: 10px; }
details.data.inline { margin-top: 0; }
pre {
  margin: 8px 0 0;
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 12px;
  line-height: 1.45;
  overflow-x: auto;
  white-space: pre;
  color: var(--text-secondary);
}

@media (max-width: 720px) {
  .hero-card { grid-template-columns: 1fr; }
  .hero-value { font-size: 44px; }
  .option { grid-template-columns: 20px 1fr 56px 36px; }
  .ladder li { grid-template-columns: 60px 1fr 48px; }
}
@media print {
  body { background: #fff; }
  .ghost, .filters { display: none; }
  details.data { display: block; }
  details.data > summary { display: none; }
  .decision { break-inside: avoid; }
}
</style>
</head>
<body>
"##;

/// The filters and the theme toggle. Nothing here carries data.
const TAIL: &str = r#"<script>
(function () {
  var items = Array.prototype.slice.call(document.querySelectorAll(".decision"));
  var chips = Array.prototype.slice.call(document.querySelectorAll(".chip"));
  var shown = document.getElementById("shown");
  var empty = document.getElementById("empty");

  function apply(mode) {
    var count = 0;
    items.forEach(function (item) {
      var keep = mode === "all"
        || (mode === "wrong" && item.getAttribute("data-wrong") === "true")
        || (mode === "split" && item.getAttribute("data-split") === "true")
        || (mode === "unsure" && item.getAttribute("data-unsure") === "true");
      item.hidden = !keep;
      if (keep) count++;
    });
    shown.textContent = count + " shown";
    empty.hidden = count !== 0;
    chips.forEach(function (chip) {
      chip.setAttribute("aria-pressed",
        String(chip.getAttribute("data-filter") === mode));
    });
  }

  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      apply(chip.getAttribute("data-filter"));
    });
  });

  var button = document.getElementById("theme");
  if (button) {
    button.addEventListener("click", function () {
      var root = document.documentElement;
      var dark = root.getAttribute("data-theme") === "dark"
        || (!root.hasAttribute("data-theme")
            && window.matchMedia("(prefers-color-scheme: dark)").matches);
      root.setAttribute("data-theme", dark ? "light" : "dark");
    });
  }
})();
</script>
</body>
</html>
"#;
