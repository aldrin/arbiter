//! What comes back: one typed answer per question, plus what the request cost.
//!
//! Confidence here is a calibrated probability from the model, not a figure it
//! was asked to report about itself. That is the point of the API: thresholds
//! on these numbers mean something, so code can act on a confident answer and
//! escalate an unconfident one.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};

/// A yes/no judgement.
///
/// The probability *is* the answer; there is no separate confidence. A value
/// near 0.5 is the model saying the state does not settle the question.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Noul {
    /// How likely the judgement is true, in `0.0..=1.0`.
    #[serde(rename = "noul")]
    pub probability: f64,
}

impl Noul {
    /// Whether the judgement is true at or above `threshold`.
    pub fn is_yes(&self, threshold: f64) -> bool {
        self.probability >= threshold
    }

    /// Whether the judgement is false at or above `threshold`.
    pub fn is_no(&self, threshold: f64) -> bool {
        1.0 - self.probability >= threshold
    }

    /// How far the judgement sits from an even split, in `0.0..=0.5`.
    ///
    /// The useful "how settled is this" number for a yes/no question, since
    /// both 0.02 and 0.98 are decisive while 0.5 is not.
    pub fn decisiveness(&self) -> f64 {
        (self.probability - 0.5).abs()
    }
}

/// A choice among named options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Choice {
    /// The option chosen, as the request labelled it.
    pub choice: String,
    /// Calibrated confidence in the choice. Absent when the provider does not
    /// report one.
    #[serde(default)]
    pub confidence: Option<f64>,
    /// The distribution over every offered option. May be empty.
    #[serde(default)]
    pub probabilities: BTreeMap<String, f64>,
}

impl Choice {
    /// Whether confidence is known and at or above `threshold`.
    ///
    /// An unreported confidence is not confident: the gate fails closed.
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence.is_some_and(|value| value >= threshold)
    }

    /// The highest-scoring option that was not chosen.
    pub fn runner_up(&self) -> Option<(&str, f64)> {
        self.probabilities
            .iter()
            .filter(|(option, _)| *option != &self.choice)
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(option, probability)| (option.as_str(), *probability))
    }

    /// How far ahead the chosen option is of the runner-up.
    ///
    /// A small margin means two options were nearly tied, which `confidence`
    /// alone does not always make obvious.
    pub fn margin(&self) -> Option<f64> {
        let chosen = self.probabilities.get(&self.choice)?;
        let (_, runner_up) = self.runner_up()?;
        Some(chosen - runner_up)
    }
}

/// A position on an ordered scale.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Score {
    /// Where on the scale the state falls, as an expected value over the
    /// levels. With three levels this ranges over `0.0..=2.0`.
    pub score: f64,
    /// Calibrated confidence in the score. Absent when not reported.
    #[serde(default)]
    pub confidence: Option<f64>,
    /// What each level index means, as the request described it.
    #[serde(default)]
    pub legend: BTreeMap<String, Value>,
    /// The distribution over level indices. May be empty.
    #[serde(default)]
    pub probabilities: BTreeMap<String, f64>,
}

impl Score {
    /// Whether confidence is known and at or above `threshold`.
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence.is_some_and(|value| value >= threshold)
    }

    /// The nearest whole level to [`Self::score`].
    ///
    /// The score is an expected value, so 1.99 means "almost entirely the
    /// level at index 2" and rounds there.
    pub fn nearest_level(&self) -> usize {
        if self.score.is_nan() || self.score < 0.0 {
            return 0;
        }
        // `as` on a rounded non-negative finite f64 is the intended cast here.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let level = self.score.round() as usize;
        level
    }

    /// What [`Self::nearest_level`] means, when the provider sent a legend.
    pub fn nearest_label(&self) -> Option<&Value> {
        self.legend.get(&self.nearest_level().to_string())
    }
}

/// One question's answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    /// A yes/no judgement.
    Noul(Noul),
    /// A choice among options.
    Choice(Choice),
    /// A position on a scale.
    Score(Score),
}

impl Answer {
    /// The wire name of this answer's type.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Noul(_) => "noul",
            Self::Choice(_) => "choice",
            Self::Score(_) => "score",
        }
    }
}

/// What a request cost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens of state and questions sent. Billing is on these.
    pub input_tokens: u64,
    /// Tokens of answers returned. Currently free on both providers.
    pub output_tokens: u64,
    /// What the request cost, in USD, when reported.
    #[serde(default)]
    pub cost: Option<f64>,
}

/// Every answer to one request.
///
/// Answers are keyed by the name the question was asked under. The typed
/// accessors — [`Self::choice`], [`Self::noul`], [`Self::score`] — fail
/// rather than return a default when a name is missing or came back as a
/// different type, so a renamed question is a loud error and not a silently
/// wrong decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Answers {
    /// The provider's identifier for this request, for correlating logs.
    #[serde(default)]
    pub id: Option<String>,
    /// The model that answered, usually more specific than the one asked for.
    pub model: String,
    /// The upstream provider that served it.
    #[serde(default)]
    pub provider: Option<String>,
    /// What the request cost.
    pub usage: Usage,
    /// Each answer, keyed by question name.
    pub answers: BTreeMap<String, Answer>,
}

impl Answers {
    /// The raw answer to `name`, whatever its type.
    pub fn get(&self, name: &str) -> Option<&Answer> {
        self.answers.get(name)
    }

    /// The names that were answered.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.answers.keys().map(String::as_str)
    }

    /// How many questions were answered.
    pub fn len(&self) -> usize {
        self.answers.len()
    }

    /// Whether nothing was answered.
    pub fn is_empty(&self) -> bool {
        self.answers.is_empty()
    }

    /// The yes/no answer to `name`.
    pub fn noul(&self, name: &str) -> Result<&Noul> {
        match self.require(name)? {
            Answer::Noul(noul) => Ok(noul),
            other => Err(Self::wrong_type(name, "noul", other)),
        }
    }

    /// The choice answer to `name`.
    pub fn choice(&self, name: &str) -> Result<&Choice> {
        match self.require(name)? {
            Answer::Choice(choice) => Ok(choice),
            other => Err(Self::wrong_type(name, "choice", other)),
        }
    }

    /// The score answer to `name`.
    pub fn score(&self, name: &str) -> Result<&Score> {
        match self.require(name)? {
            Answer::Score(score) => Ok(score),
            other => Err(Self::wrong_type(name, "score", other)),
        }
    }

    /// The answer to `name`, or an error naming what did come back.
    fn require(&self, name: &str) -> Result<&Answer> {
        self.answers.get(name).ok_or_else(|| Error::NoSuchAnswer {
            name: name.to_owned(),
            answered: self.answers.keys().cloned().collect(),
        })
    }

    /// An [`Error::AnswerType`] for `name`.
    fn wrong_type(name: &str, wanted: &'static str, got: &Answer) -> Error {
        Error::AnswerType {
            name: name.to_owned(),
            wanted,
            got: got.type_name(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Answers, Choice, Noul, Score};
    use crate::error::Error;

    /// The response documented in the API reference.
    const DOCUMENTED: &str = r#"{
      "answers": {
        "is_bug": { "noul": 0.96, "type": "noul" },
        "team": {
          "choice": "payments",
          "confidence": 0.75,
          "probabilities": { "account": 0, "frontend": 0.16, "payments": 0.84 },
          "type": "choice"
        },
        "urgency": {
          "confidence": 0.99,
          "legend": { "0": "Can wait for the next release", "1": "Should be fixed this week", "2": "Blocking revenue right now" },
          "probabilities": { "0": 0, "1": 0.01, "2": 0.99 },
          "score": 1.99,
          "type": "score"
        }
      },
      "id": "gen-dec-1789738314-X5e5eKGQdvR9rblyX250",
      "model": "typesafe/jev-1.13-20260917",
      "provider": "TypeSafe",
      "usage": { "cost": 0.000019992, "input_tokens": 476, "output_tokens": 70 }
    }"#;

    fn documented() -> Answers {
        serde_json::from_str(DOCUMENTED).expect("the documented response must parse")
    }

    #[test]
    fn parses_the_documented_response() {
        let answers = documented();
        assert_eq!(answers.model, "typesafe/jev-1.13-20260917");
        assert_eq!(answers.provider.as_deref(), Some("TypeSafe"));
        assert_eq!(
            answers.id.as_deref(),
            Some("gen-dec-1789738314-X5e5eKGQdvR9rblyX250")
        );
        assert_eq!(answers.usage.input_tokens, 476);
        assert_eq!(answers.usage.output_tokens, 70);
        assert_eq!(answers.usage.cost, Some(0.000_019_992));
        assert_eq!(answers.len(), 3);
    }

    #[test]
    fn reads_each_primitive_with_its_own_accessor() {
        let answers = documented();

        assert!((answers.noul("is_bug").expect("noul").probability - 0.96).abs() < 1e-9);

        let team = answers.choice("team").expect("choice");
        assert_eq!(team.choice, "payments");
        assert_eq!(team.confidence, Some(0.75));
        assert!((team.probabilities["frontend"] - 0.16).abs() < 1e-9);

        let urgency = answers.score("urgency").expect("score");
        assert!((urgency.score - 1.99).abs() < 1e-9);
        assert_eq!(urgency.nearest_level(), 2);
        assert_eq!(
            urgency.nearest_label().and_then(|v| v.as_str()),
            Some("Blocking revenue right now")
        );
    }

    #[test]
    fn asking_for_the_wrong_type_is_an_error() {
        match documented().noul("team") {
            Err(Error::AnswerType { name, wanted, got }) => {
                assert_eq!((name.as_str(), wanted, got), ("team", "noul", "choice"));
            }
            other => panic!("expected AnswerType, got {other:?}"),
        }
    }

    #[test]
    fn asking_for_an_unanswered_name_lists_what_was_answered() {
        match documented().choice("owner") {
            Err(Error::NoSuchAnswer { name, answered }) => {
                assert_eq!(name, "owner");
                assert_eq!(answered, ["is_bug", "team", "urgency"]);
            }
            other => panic!("expected NoSuchAnswer, got {other:?}"),
        }
    }

    #[test]
    fn choice_confidence_fails_closed_when_unreported() {
        let bare = Choice {
            choice: "payments".into(),
            confidence: None,
            probabilities: std::collections::BTreeMap::new(),
        };
        assert!(
            !bare.is_confident(0.0),
            "unknown confidence is not confident"
        );
    }

    #[test]
    fn choice_reports_its_runner_up_and_margin() {
        let team = documented().choice("team").expect("choice").clone();
        assert_eq!(team.runner_up(), Some(("frontend", 0.16)));
        assert!((team.margin().expect("margin") - 0.68).abs() < 1e-9);
    }

    #[test]
    fn a_single_option_distribution_has_no_runner_up() {
        let only = Choice {
            choice: "payments".into(),
            confidence: Some(1.0),
            probabilities: [("payments".to_owned(), 1.0)].into_iter().collect(),
        };
        assert!(only.runner_up().is_none());
        assert!(only.margin().is_none());
    }

    #[test]
    fn noul_reads_from_either_side() {
        let likely = Noul { probability: 0.96 };
        assert!(likely.is_yes(0.9));
        assert!(!likely.is_no(0.5));
        assert!((likely.decisiveness() - 0.46).abs() < 1e-9);

        let unlikely = Noul { probability: 0.04 };
        assert!(unlikely.is_no(0.9));
        assert!(!unlikely.is_yes(0.5));

        // An even split is decisive from neither side.
        let split = Noul { probability: 0.5 };
        assert!(!split.is_yes(0.6));
        assert!(!split.is_no(0.6));
        assert!(split.decisiveness().abs() < f64::EPSILON);
    }

    #[test]
    fn score_rounding_survives_odd_values() {
        let level = |score| Score {
            score,
            confidence: None,
            legend: std::collections::BTreeMap::new(),
            probabilities: std::collections::BTreeMap::new(),
        };
        assert_eq!(level(0.0).nearest_level(), 0);
        assert_eq!(level(1.49).nearest_level(), 1);
        assert_eq!(level(1.5).nearest_level(), 2);
        // Nonsense from a provider must not panic or wrap around.
        assert_eq!(level(-3.0).nearest_level(), 0);
        assert_eq!(level(f64::NAN).nearest_level(), 0);
    }

    #[test]
    fn optional_response_fields_may_be_absent() {
        // Only model, answers and usage are required by the schema.
        let minimal = r#"{
          "model": "typesafe/jev-1.13",
          "usage": { "input_tokens": 10, "output_tokens": 2 },
          "answers": { "q": { "type": "choice", "choice": "a" } }
        }"#;
        let answers: Answers = serde_json::from_str(minimal).expect("minimal response must parse");
        assert!(answers.id.is_none());
        assert!(answers.provider.is_none());
        assert!(answers.usage.cost.is_none());

        let q = answers.choice("q").expect("choice");
        assert_eq!(q.choice, "a");
        assert!(q.confidence.is_none());
        assert!(q.probabilities.is_empty());
    }
}
