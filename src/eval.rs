//! The record of an evaluation: many questions, each asked more than once.
//!
//! Nothing here knows what is being evaluated. A record is a repeated
//! [`choice`](crate::Request::choice) over questions with a known expected
//! answer — the shape any such evaluation has — so a harness that produces one
//! and a view that reads one need not know about each other.
//!
//! The file is the contract between the two, which is why it is a declared
//! schema rather than whatever a particular view happened to need. It holds
//! everything observed: every attempt at every question, and both halves of
//! each exchange. A new view can be built from an old record without paying
//! for inference again.
//!
//! Derived numbers live here too, computed from the record rather than stored
//! in it: a statistic that can be recomputed should not be able to disagree
//! with the data it came from.
//!
//! ```
//! use arbiter::eval::{Attempt, Question, Stats};
//!
//! let question = Question {
//!     index: 0,
//!     text: "Which team owns checkout?".to_owned(),
//!     source: None,
//!     options: [("A".to_owned(), "payments".to_owned())].into_iter().collect(),
//!     expected: "A".to_owned(),
//!     payload: serde_json::Value::Null,
//!     attempts: vec![Attempt {
//!         ordinal: 1,
//!         chosen: Some("A".to_owned()),
//!         correct: Some(true),
//!         ..Attempt::failed(1, 0, "unused".to_owned())
//!     }],
//! };
//! assert_eq!(question.correct_count(), 1);
//! assert!(question.is_unanimous());
//! ```

// Counting turns integers into ratios all over this module. Every count is a
// sample size, far below where an f64 loses integers.
#![allow(clippy::cast_precision_loss)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

// The run record: what `run` writes and `render` reads.
//
// The file is the contract between the two modes, which is why it is a
// declared schema rather than whatever the renderer happened to need. It
// holds everything observed — every attempt at every question, both halves of
// each exchange — so a new view can be built from an old run without paying
// for inference again.
//
// Derived numbers live here too, computed from the record rather than stored
// in it: a statistic that can be recomputed should not be able to disagree
// with the data it came from.

/// The schema this build writes and can read.
///
/// Bumped when a change would make an older file parse into something that
/// means something different. `render` refuses a version it does not know,
/// because silently rendering a misread file is worse than a clear error.
pub const SCHEMA: &str = "arbiter.choice-eval/1";

/// One whole run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// Always [`SCHEMA`] for a file this build wrote.
    pub schema: String,
    /// When the run started, in milliseconds since the Unix epoch.
    pub started_unix_ms: u64,
    /// What was evaluated, as a slug.
    pub benchmark: String,
    /// Who served the requests.
    pub provider: String,
    /// The model asked for. What answered is in each attempt's response.
    pub model: String,
    /// The sampling seed, so the same questions can be asked again.
    pub seed: u64,
    /// How many times each question was asked.
    pub attempts: usize,
    /// Requests in flight at once.
    pub concurrency: usize,
    /// Questions asked in this run.
    pub asked: usize,
    /// Usable questions the source had to offer.
    pub available: usize,
    /// Questions the source offered but could not be used.
    pub skipped: usize,
    /// Attempts that produced no decision at all.
    pub failures: usize,
    /// How long the run took, end to end.
    pub wall_ms: u64,
    /// What the run spent.
    pub usage: Usage,
    /// Every question, in the order it was asked.
    pub questions: Vec<Question>,
}

/// What a run spent.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct Usage {
    /// Requests that were answered.
    pub calls: usize,
    /// Tokens of state and questions sent. Billing is on these.
    pub input_tokens: u64,
    /// Tokens of answers returned.
    pub output_tokens: u64,
    /// Present only when the provider reports it; TypeSafe does not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<f64>,
}

impl Usage {
    /// Add what one answered request cost.
    pub fn record(&mut self, spent: &crate::Usage) {
        self.calls += 1;
        self.input_tokens += spent.input_tokens;
        self.output_tokens += spent.output_tokens;
        if let Some(cost) = spent.cost {
            *self.cost.get_or_insert(0.0) += cost;
        }
    }

    /// What the run cost, and whether the provider said so.
    ///
    /// `usd_per_million_input_tokens` is only consulted when it did not —
    /// TypeSafe reports no cost, OpenRouter does. The price is the caller's to
    /// supply: it drifts, and a library that baked one in would go stale
    /// between releases.
    pub fn dollars(&self, usd_per_million_input_tokens: f64) -> (f64, bool) {
        match self.cost {
            Some(cost) => (cost, true),
            None => (
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "a token count far below where an f64 loses integers"
                )]
                {
                    self.input_tokens as f64 * usd_per_million_input_tokens / 1_000_000.0
                },
                false,
            ),
        }
    }
}

/// One question, and every attempt at it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    /// Position in the sample, so a decision can be referred to.
    pub index: usize,
    /// The question as the dataset words it.
    #[serde(rename = "question")]
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Where the question came from, when its source says.
    pub source: Option<String>,
    /// Option label to its text.
    pub options: BTreeMap<String, String>,
    /// The option label the source says is correct.
    pub expected: String,
    /// The body posted, stored once: it is identical for every attempt, so
    /// repeating it per attempt would multiply the file for nothing.
    pub payload: Value,
    /// Every attempt at this question, in order.
    pub attempts: Vec<Attempt>,
}

impl Question {
    /// Attempts that produced a decision.
    pub fn decided(&self) -> impl Iterator<Item = &Attempt> {
        self.attempts.iter().filter(|a| a.chosen.is_some())
    }

    /// How many attempts chose the correct label.
    pub fn correct_count(&self) -> usize {
        self.decided().filter(|a| a.correct == Some(true)).count()
    }

    /// How many attempts produced a decision at all.
    pub fn decided_count(&self) -> usize {
        self.decided().count()
    }

    /// Whether every decided attempt chose the same label.
    ///
    /// Vacuously true for fewer than two decisions — there is nothing to
    /// disagree with — so callers that care about flakiness should pair this
    /// with [`Self::decided_count`].
    pub fn is_unanimous(&self) -> bool {
        let mut chosen = self.decided().filter_map(|a| a.chosen.as_deref());
        let Some(first) = chosen.next() else {
            return true;
        };
        chosen.all(|label| label == first)
    }

    /// The label chosen most often, and how many attempts chose it.
    ///
    /// Ties break toward the earlier label, which is arbitrary but stable;
    /// a tie is itself the interesting fact and the report says so.
    pub fn majority(&self) -> Option<(String, usize)> {
        let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
        for attempt in self.decided() {
            if let Some(label) = attempt.chosen.as_deref() {
                *counts.entry(label).or_default() += 1;
            }
        }
        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(label, count)| (label.to_owned(), count))
    }

    /// Whether the majority label is the correct one.
    pub fn majority_is_correct(&self) -> bool {
        self.majority()
            .is_some_and(|(label, _)| label == self.expected)
    }

    /// Mean confidence across the decided attempts.
    pub fn mean_confidence(&self) -> Option<f64> {
        let values: Vec<f64> = self.decided().filter_map(|a| a.confidence).collect();
        if values.is_empty() {
            return None;
        }
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

impl Attempt {
    /// An attempt that decided nothing, with the reason.
    ///
    /// The shape a harness reaches for when a request fails: a failure is a
    /// fact about the run, so it is recorded rather than dropped.
    pub fn failed(ordinal: usize, latency_ms: u64, why: String) -> Self {
        Self {
            ordinal,
            chosen: None,
            correct: None,
            confidence: None,
            margin: None,
            probabilities: BTreeMap::new(),
            latency_ms,
            response: None,
            error: Some(why),
        }
    }
}

/// One ask, and what came back.
///
/// An attempt either decided something — `chosen` and the fields beside it —
/// or failed, in which case `error` says why. Failures are kept rather than
/// dropped: "this question failed twice out of five" is a fact about the run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attempt {
    /// 1-based, in the order the attempts were issued.
    #[serde(rename = "attempt")]
    pub ordinal: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// The option label chosen, absent when the attempt decided nothing.
    pub chosen: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Whether [`Self::chosen`] matches the expected label.
    pub correct: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    /// Calibrated confidence in the choice, when the provider reported one.
    pub confidence: Option<f64>,
    /// How far ahead the chosen label was of the runner-up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub margin: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    /// The probability given to every option offered.
    pub probabilities: BTreeMap<String, f64>,
    /// How long this one ask took.
    pub latency_ms: u64,
    /// The response body: JSON when it parsed, a JSON string when it did not,
    /// absent when the request never got one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response: Option<Value>,
    /// Why this attempt produced no decision.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ----------------------------------------------------------------- derived

/// Everything the report shows about a run as a whole.
///
/// Recomputed from the record every time rather than stored, so it cannot
/// drift from the attempts it describes.
#[derive(Debug, Clone)]
pub struct Stats {
    /// Questions in the record.
    pub questions: usize,
    /// Attempts requested per question.
    pub attempts_each: usize,
    /// Attempts that produced a decision.
    pub decided_attempts: usize,
    /// Of those, how many chose the correct label.
    pub correct_attempts: usize,
    /// Questions with at least one decision.
    pub decided_questions: usize,
    /// `pass^k` for every `k` the run can support, lowest first.
    pub pass_hat: Vec<(usize, f64)>,
    /// Questions where every decided attempt chose the same label.
    pub unanimous: usize,
    /// Questions where every decided attempt chose the *correct* label.
    pub unanimous_correct: usize,
    /// Questions whose most-chosen label is the correct one.
    pub majority_correct: usize,
    /// Mean confidence across every decided attempt.
    pub mean_confidence: Option<f64>,
    /// Median latency across every attempt, failures included.
    pub median_latency_ms: u64,
}

impl Stats {
    /// Everything derivable, computed once from `run`.
    pub fn of(run: &Run) -> Self {
        let questions = run.questions.len();
        let decided_attempts: usize = run.questions.iter().map(Question::decided_count).sum();
        let correct_attempts: usize = run.questions.iter().map(Question::correct_count).sum();

        let confidences: Vec<f64> = run
            .questions
            .iter()
            .flat_map(|q| q.decided().filter_map(|a| a.confidence))
            .collect();
        let mean_confidence = (!confidences.is_empty())
            .then(|| confidences.iter().sum::<f64>() / confidences.len() as f64);

        let mut latencies: Vec<u64> = run
            .questions
            .iter()
            .flat_map(|q| q.attempts.iter().map(|a| a.latency_ms))
            .collect();
        latencies.sort_unstable();

        // pass^k only means something up to the number of attempts every
        // question actually got, so cap at the smallest.
        let floor = run
            .questions
            .iter()
            .map(Question::decided_count)
            .min()
            .unwrap_or(0);
        let pass_hat = (1..=floor.min(run.attempts))
            .map(|k| (k, pass_hat_k(&run.questions, k)))
            .collect();

        Self {
            questions,
            attempts_each: run.attempts,
            decided_attempts,
            correct_attempts,
            decided_questions: run
                .questions
                .iter()
                .filter(|q| q.decided_count() > 0)
                .count(),
            pass_hat,
            unanimous: run
                .questions
                .iter()
                .filter(|q| q.decided_count() > 1 && q.is_unanimous())
                .count(),
            unanimous_correct: run
                .questions
                .iter()
                .filter(|q| q.decided_count() > 0 && q.is_unanimous() && q.majority_is_correct())
                .count(),
            majority_correct: run
                .questions
                .iter()
                .filter(|q| q.majority_is_correct())
                .count(),
            mean_confidence,
            median_latency_ms: latencies
                .get(latencies.len() / 2)
                .copied()
                .unwrap_or_default(),
        }
    }

    /// `pass^1`, which is accuracy over every attempt.
    pub fn accuracy(&self) -> f64 {
        ratio(self.correct_attempts, self.decided_attempts)
    }

    /// The highest `k` the run supports.
    pub fn deepest_k(&self) -> Option<(usize, f64)> {
        self.pass_hat.last().copied()
    }
}

/// `pass^k` over `questions`: the chance that `k` attempts, drawn without
/// replacement, are *all* correct.
///
/// Note the caret, not an at-sign. `pass@k` asks whether *any* of `k` attempts
/// succeeds and rewards a lucky guess; `pass^k` asks whether *every* one does,
/// which is the question worth asking of a decision you intend to act on
/// unsupervised. The estimator is the hypergeometric one — `C(c, k) / C(n, k)`
/// per question, averaged — rather than `(c/n)^k`, which is biased.
pub fn pass_hat_k(questions: &[Question], k: usize) -> f64 {
    let each: Vec<f64> = questions
        .iter()
        .filter_map(|question| {
            all_correct_probability(question.correct_count(), question.decided_count(), k)
        })
        .collect();
    if each.is_empty() {
        return 0.0;
    }
    each.iter().sum::<f64>() / each.len() as f64
}

/// `C(correct, k) / C(attempts, k)`, computed as a running product so nothing
/// overflows on the way to a ratio.
///
/// `None` when the question cannot answer for `k` at all, which keeps a
/// question that failed its requests out of the average instead of counting
/// as a zero.
pub fn all_correct_probability(correct: usize, attempts: usize, k: usize) -> Option<f64> {
    if k == 0 || k > attempts {
        return None;
    }
    if correct < k {
        return Some(0.0);
    }
    Some(
        (0..k)
            .map(|i| (correct - i) as f64 / (attempts - i) as f64)
            .product(),
    )
}

/// `part / whole`, and zero rather than NaN when there is nothing to divide.
pub fn ratio(part: usize, whole: usize) -> f64 {
    if whole == 0 {
        return 0.0;
    }
    part as f64 / whole as f64
}

/// The Wilson score interval at 95%, for a proportion.
///
/// Preferred over the textbook normal approximation, which misbehaves near 0
/// and 1 and can report a bound outside `0..=1`.
pub fn wilson(hits: usize, total: usize) -> (f64, f64) {
    /// The two-sided 95% normal quantile.
    pub const Z: f64 = 1.959_963_984_540_054;

    if total == 0 {
        return (0.0, 0.0);
    }
    let n = total as f64;
    let p = hits as f64 / n;
    let denominator = 1.0 + Z * Z / n;
    let centre = p + Z * Z / (2.0 * n);
    let spread = Z * ((p * (1.0 - p) / n) + Z * Z / (4.0 * n * n)).sqrt();
    (
        ((centre - spread) / denominator).max(0.0),
        ((centre + spread) / denominator).min(1.0),
    )
}

/// A UTC timestamp like `2026-10-04 21:07:13Z`, from Unix milliseconds.
///
/// Written out rather than taken from a date crate: the library has no date
/// dependency and one timestamp in a header is not worth adding one.
pub fn format_unix_ms(ms: u64) -> String {
    let seconds = ms / 1000;
    let (days, rest) = (seconds / 86_400, seconds % 86_400);
    let (hour, minute, second) = (rest / 3600, (rest % 3600) / 60, rest % 60);

    // Howard Hinnant's civil-from-days, shifted to a 0000-03-01 era so leap
    // years fall at the end of the cycle and need no special case.
    let z = i64::try_from(days).unwrap_or(i64::MAX / 2) + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z.rem_euclid(146_097);
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let shifted_month = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * shifted_month + 2) / 5 + 1;
    let month = if shifted_month < 10 {
        shifted_month + 3
    } else {
        shifted_month - 9
    };
    let year = year + i64::from(month <= 2);

    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02}:{second:02}Z")
}

#[cfg(test)]
mod tests {
    use super::{all_correct_probability, format_unix_ms};

    #[test]
    fn pass_hat_k_is_one_when_every_attempt_is_correct() {
        assert_eq!(all_correct_probability(5, 5, 5), Some(1.0));
        assert_eq!(all_correct_probability(5, 5, 1), Some(1.0));
    }

    #[test]
    fn pass_hat_k_is_zero_when_too_few_are_correct() {
        // Four of five correct cannot give five-in-a-row.
        assert_eq!(all_correct_probability(4, 5, 5), Some(0.0));
        assert_eq!(all_correct_probability(0, 5, 1), Some(0.0));
    }

    #[test]
    fn pass_hat_k_interpolates_between() {
        // C(4,2)/C(5,2) = 6/10.
        let p = all_correct_probability(4, 5, 2).expect("defined");
        assert!((p - 0.6).abs() < 1e-12, "{p}");
        // C(3,2)/C(5,2) = 3/10.
        let p = all_correct_probability(3, 5, 2).expect("defined");
        assert!((p - 0.3).abs() < 1e-12, "{p}");
    }

    #[test]
    fn pass_hat_k_is_undefined_past_the_attempts_made() {
        assert!(all_correct_probability(2, 2, 3).is_none());
        assert!(all_correct_probability(2, 2, 0).is_none());
    }

    #[test]
    fn timestamps_render_as_utc() {
        assert_eq!(format_unix_ms(0), "1970-01-01 00:00:00Z");
        assert_eq!(format_unix_ms(1_000_000_000_000), "2001-09-09 01:46:40Z");
        // A leap day, which the era shift has to get right.
        assert_eq!(format_unix_ms(1_583_020_800_000), "2020-03-01 00:00:00Z");
    }
}
