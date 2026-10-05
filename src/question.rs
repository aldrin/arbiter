//! What to ask: one shared state and a battery of named, typed questions.
//!
//! The API takes many questions over a single [`state`](Request::new) and
//! evaluates them independently and in parallel, so decomposing a broad
//! judgement into narrow atomic ones costs no extra round trip. TypeSafe's
//! guidance is to prefer many specific questions over one broad one, then
//! combine the answers in your own code.
//!
//! Three primitives, each with its own shape of guidance:
//!
//! | Primitive | Asks | Answered by |
//! |---|---|---|
//! | [`Request::noul`] | a yes/no judgement | a probability |
//! | [`Request::choice`] | which of these options | one option and a distribution |
//! | [`Request::score`] | where on this ordered scale | a point on the scale |

use serde::Serialize;
use serde::ser::SerializeMap as _;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::provider::Provider;

/// Guidance attached to a question: a string for the simple case, or an object
/// or array when the question needs named context your code can swap out.
pub type Guidance = Value;

/// The name a one-question request asks under.
///
/// Answers come back keyed by name, so a front end that asks exactly one
/// question still has to pick one and look the answer up under it. Every
/// front end in this crate uses this, which is why it lives here rather than
/// as a private constant in each: a lookup under the wrong name is a
/// [`NoSuchAnswer`](crate::Error::NoSuchAnswer) at runtime and nothing
/// earlier.
pub const SINGLE_QUESTION_NAME: &str = "answer";

/// One question in a [`Request`].
///
/// The wire shape falls out of the derive: `type` from the tag, then the
/// fields in declaration order. Only two things need help — a choice's
/// options must stay in the order the caller gave them, and a noul's criteria
/// must be omitted entirely rather than half-filled when undescribed.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
    /// A yes/no judgement, answered by a probability.
    Noul {
        /// What is being asked.
        instructions: Guidance,
        /// What `true` and what `false` each mean. Optional, but describing
        /// both sides sharpens the judgement.
        //
        // The schema requires both sides when criteria is present, so an
        // undescribed noul omits the key rather than sending one side.
        #[serde(skip_serializing_if = "Option::is_none")]
        criteria: Option<NoulCriteria>,
    },
    /// A choice among named options, answered by one option and a
    /// distribution over all of them.
    Choice {
        /// What is being asked.
        instructions: Guidance,
        /// Each option and what it covers, in the order given.
        #[serde(serialize_with = "ordered")]
        criteria: Vec<(String, Guidance)>,
    },
    /// A position on an ordered scale, answered by a point on that scale.
    Score {
        /// What is being asked.
        instructions: Guidance,
        /// The levels, lowest first.
        criteria: Vec<Guidance>,
    },
}

/// What each side of a [`Question::Noul`] means.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NoulCriteria {
    /// What makes the answer true.
    #[serde(rename = "true")]
    pub when_true: Guidance,
    /// What makes the answer false.
    #[serde(rename = "false")]
    pub when_false: Guidance,
}

impl Question {
    /// The wire name of this question's type.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Noul { .. } => "noul",
            Self::Choice { .. } => "choice",
            Self::Score { .. } => "score",
        }
    }
}

/// Serializes `(name, value)` pairs as a JSON object in the order given.
///
/// `serde_json::Map` sorts its keys unless built with `preserve_order`, and
/// the order is worth keeping: for a choice's options it is the order the
/// caller declared and the order their own constants are usually listed in,
/// and for a batch it is the order the questions were asked.
fn ordered<K, V, S>(pairs: &[(K, V)], serializer: S) -> std::result::Result<S::Ok, S::Error>
where
    K: AsRef<str>,
    V: Serialize,
    S: serde::Serializer,
{
    let mut map = serializer.serialize_map(Some(pairs.len()))?;
    for (key, value) in pairs {
        map.serialize_entry(key.as_ref(), value)?;
    }
    map.end()
}

/// `(name, value)` pairs as a value that serializes in order.
///
/// The wrapper exists because [`ordered`] is reachable from a `serialize_with`
/// attribute but not from `serde_json::to_value`, which needs something that
/// implements [`Serialize`].
struct Ordered<'a, V>(&'a [(String, V)]);

impl<V: Serialize> Serialize for Ordered<'_, V> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        ordered(self.0, serializer)
    }
}

/// A batch of questions about one state.
///
/// ```
/// use arbiter::Request;
///
/// let request = Request::new(serde_json::json!({
///     "ticket": "My checkout page shows a blank screen after I click Pay.",
///     "customer_tier": "enterprise",
/// }))
/// .choice(
///     "team",
///     "Which team should own this ticket?",
///     [
///         ("payments", "Checkout, billing, or payment processing issues."),
///         ("frontend", "Rendering, layout, or browser compatibility issues."),
///         ("account", "Login, permissions, or profile issues."),
///     ],
/// )
/// .noul_described(
///     "is_bug",
///     "Is the customer reporting a software defect?",
///     "The customer describes broken or unexpected product behavior.",
///     "The customer is asking a question or requesting a feature.",
/// )
/// .score(
///     "urgency",
///     "How urgent is this ticket?",
///     [
///         "Can wait for the next release",
///         "Should be fixed this week",
///         "Blocking revenue right now",
///     ],
/// );
///
/// assert_eq!(request.len(), 3);
/// assert!(request.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    state: Value,
    questions: Vec<(String, Question)>,
    model: Option<String>,
    session_id: Option<String>,
    user: Option<String>,
}

impl Request {
    /// A request about `state`, with no questions yet.
    ///
    /// `state` is the context to evaluate: a plain string, or a JSON object or
    /// array. Send only what the questions need — a question pointed at a
    /// named field with backticks, as in ``"Does `ticket.body` ask for a
    /// refund?"``, is sharper than one pointed at everything.
    pub fn new(state: impl Into<Value>) -> Self {
        Self {
            state: state.into(),
            questions: Vec::new(),
            model: None,
            session_id: None,
            user: None,
        }
    }

    /// Ask a yes/no question.
    #[must_use]
    pub fn noul(mut self, name: impl Into<String>, instructions: impl Into<Guidance>) -> Self {
        self.questions.push((
            name.into(),
            Question::Noul {
                instructions: instructions.into(),
                criteria: None,
            },
        ));
        self
    }

    /// Ask a yes/no question, saying what each side means.
    #[must_use]
    pub fn noul_described(
        mut self,
        name: impl Into<String>,
        instructions: impl Into<Guidance>,
        when_true: impl Into<Guidance>,
        when_false: impl Into<Guidance>,
    ) -> Self {
        self.questions.push((
            name.into(),
            Question::Noul {
                instructions: instructions.into(),
                criteria: Some(NoulCriteria {
                    when_true: when_true.into(),
                    when_false: when_false.into(),
                }),
            },
        ));
        self
    }

    /// Ask a yes/no question, describing both sides when both are given.
    ///
    /// The shape every front end needs: guidance arrives as two independent
    /// optional values, and the schema takes both sides or neither. A lone
    /// side is dropped, because half a contrast is not guidance.
    #[must_use]
    pub fn noul_maybe_described(
        self,
        name: impl Into<String>,
        instructions: impl Into<Guidance>,
        when_true: Option<impl Into<Guidance>>,
        when_false: Option<impl Into<Guidance>>,
    ) -> Self {
        match (when_true, when_false) {
            (Some(yes), Some(no)) => self.noul_described(name, instructions, yes, no),
            _ => self.noul(name, instructions),
        }
    }

    /// Ask which of `options` applies, each given as `(label, description)`.
    ///
    /// Descriptions are what make a choice reliable. Contrastive ones — what
    /// an option covers and what belongs to a different option — separate
    /// options that would otherwise blur.
    #[must_use]
    pub fn choice<L, D>(
        mut self,
        name: impl Into<String>,
        instructions: impl Into<Guidance>,
        options: impl IntoIterator<Item = (L, D)>,
    ) -> Self
    where
        L: Into<String>,
        D: Into<Guidance>,
    {
        self.questions.push((
            name.into(),
            Question::Choice {
                instructions: instructions.into(),
                criteria: options
                    .into_iter()
                    .map(|(label, description)| (label.into(), description.into()))
                    .collect(),
            },
        ));
        self
    }

    /// Ask which of `labels` applies, with no descriptions.
    ///
    /// Fine when the labels speak for themselves; prefer [`Self::choice`]
    /// otherwise.
    #[must_use]
    pub fn choice_bare<L: Into<String>>(
        self,
        name: impl Into<String>,
        instructions: impl Into<Guidance>,
        labels: impl IntoIterator<Item = L>,
    ) -> Self {
        self.choice(
            name,
            instructions,
            labels.into_iter().map(|label| (label, Value::Null)),
        )
    }

    /// Ask where on an ordered scale the state falls. Lowest level first.
    #[must_use]
    pub fn score<D: Into<Guidance>>(
        mut self,
        name: impl Into<String>,
        instructions: impl Into<Guidance>,
        levels: impl IntoIterator<Item = D>,
    ) -> Self {
        self.questions.push((
            name.into(),
            Question::Score {
                instructions: instructions.into(),
                criteria: levels.into_iter().map(Into::into).collect(),
            },
        ));
        self
    }

    /// Ask `question` under `name`, for a question built elsewhere.
    #[must_use]
    pub fn ask(mut self, name: impl Into<String>, question: Question) -> Self {
        self.questions.push((name.into(), question));
        self
    }

    /// Decide with this model instead of the client's.
    #[must_use]
    pub fn model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Group this request with others under one identifier, for
    /// observability.
    ///
    /// An OpenRouter extension: it is sent to that provider and withheld from
    /// TypeSafe, which documents no such field. Never shown to the model —
    /// only [`Self::new`]'s state is judged. Capped at 256 characters.
    #[must_use]
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Attribute this request to an end user.
    ///
    /// An OpenRouter extension, like [`Self::session_id`], and withheld from
    /// TypeSafe. Capped at 256 characters.
    #[must_use]
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// How many questions this request asks.
    pub fn len(&self) -> usize {
        self.questions.len()
    }

    /// Whether this request asks nothing.
    pub fn is_empty(&self) -> bool {
        self.questions.is_empty()
    }

    /// The questions asked, in the order they were added.
    pub fn questions(&self) -> &[(String, Question)] {
        &self.questions
    }

    /// The model this request overrides to, if any.
    pub fn model_override(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// Check the request is answerable before it is sent.
    ///
    /// Every client call validates first, so a malformed request costs
    /// nothing.
    pub fn validate(&self) -> Result<()> {
        if self.questions.is_empty() {
            return Err(Error::request("a request must ask at least one question"));
        }

        for (index, (name, question)) in self.questions.iter().enumerate() {
            if name.trim().is_empty() {
                return Err(Error::request(format!("question {index} has no name")));
            }

            // Answers come back keyed by name, so duplicates would make one
            // answer unreachable.
            if let Some(twin) = self.questions[..index]
                .iter()
                .position(|(earlier, _)| earlier == name)
            {
                return Err(Error::request(format!(
                    "questions {twin} and {index} are both named {name:?}"
                )));
            }

            match question {
                Question::Noul { .. } => {}
                Question::Choice { criteria, .. } => {
                    if criteria.len() < 2 {
                        return Err(Error::request(format!(
                            "choice {name:?} offers {} option(s); a choice needs at least two",
                            criteria.len()
                        )));
                    }
                    for (position, (label, _)) in criteria.iter().enumerate() {
                        if label.trim().is_empty() {
                            return Err(Error::request(format!(
                                "choice {name:?} has an unlabelled option at {position}"
                            )));
                        }
                        if criteria[..position]
                            .iter()
                            .any(|(earlier, _)| earlier == label)
                        {
                            return Err(Error::request(format!(
                                "choice {name:?} offers {label:?} twice"
                            )));
                        }
                    }
                }
                Question::Score { criteria, .. } => {
                    if criteria.len() < 2 {
                        return Err(Error::request(format!(
                            "score {name:?} defines {} level(s); a scale needs at least two",
                            criteria.len()
                        )));
                    }
                }
            }
        }

        // Characters, not bytes: a 200-character identifier with an accent in
        // it is within the documented cap, and `str::len` would reject it.
        if self
            .session_id
            .as_ref()
            .is_some_and(|id| id.chars().count() > 256)
        {
            return Err(Error::request("session_id exceeds 256 characters"));
        }
        if self
            .user
            .as_ref()
            .is_some_and(|user| user.chars().count() > 256)
        {
            return Err(Error::request("user exceeds 256 characters"));
        }
        Ok(())
    }

    /// The JSON body for this request, deciding with `default_model` unless
    /// the request overrides it.
    ///
    /// `provider` decides whether the attribution fields are included: they
    /// are OpenRouter extensions, and TypeSafe documents the body as `state`,
    /// `model` and `questions` only.
    ///
    /// Consumes the request: this is the last thing done with one, and the
    /// state can be as large as the caller made it.
    pub(crate) fn into_body(self, default_model: &str, provider: Provider) -> Value {
        let questions = serde_json::to_value(Ordered(&self.questions))
            // Infallible: a `Guidance` is a `Value`, which cannot hold a
            // non-finite float, and every key is a `String`. Panicking beats
            // silently posting a request with no questions in it.
            .expect("a question built from JSON values must serialize");

        let mut body = serde_json::Map::new();
        body.insert(
            "model".to_owned(),
            Value::String(self.model.unwrap_or_else(|| default_model.to_owned())),
        );
        body.insert("state".to_owned(), self.state);
        body.insert("questions".to_owned(), questions);
        if provider.accepts_attribution() {
            if let Some(session_id) = self.session_id {
                body.insert("session_id".to_owned(), Value::String(session_id));
            }
            if let Some(user) = self.user {
                body.insert("user".to_owned(), Value::String(user));
            }
        } else if self.session_id.is_some() || self.user.is_some() {
            tracing::debug!(
                provider = %provider,
                "dropping session_id/user: this provider does not accept them"
            );
        }
        Value::Object(body)
    }
}

#[cfg(test)]
mod tests {
    use super::Request;
    use crate::provider::Provider;
    use serde_json::json;

    fn ticket() -> Request {
        Request::new(json!({"ticket": "blank screen after Pay"}))
    }

    #[test]
    fn a_request_must_ask_something() {
        assert!(ticket().validate().is_err());
        assert!(
            ticket()
                .noul("is_bug", "Is this a defect?")
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn duplicate_question_names_are_rejected() {
        // Answers are keyed by name, so a duplicate would be unreachable.
        let request = ticket().noul("same", "First?").noul("same", "Second?");
        assert!(request.validate().is_err());
    }

    #[test]
    fn a_choice_needs_at_least_two_options() {
        let one = ticket().choice("team", "Who owns it?", [("payments", "billing")]);
        assert!(one.validate().is_err());

        let two = ticket().choice(
            "team",
            "Who owns it?",
            [("payments", "billing"), ("frontend", "layout")],
        );
        assert!(two.validate().is_ok());
    }

    #[test]
    fn a_choice_rejects_repeated_labels() {
        let request = ticket().choice_bare("team", "Who?", ["payments", "payments"]);
        assert!(request.validate().is_err());
    }

    #[test]
    fn a_score_needs_at_least_two_levels() {
        assert!(
            ticket()
                .score("urgency", "How urgent?", ["only one"])
                .validate()
                .is_err()
        );
        assert!(
            ticket()
                .score("urgency", "How urgent?", ["low", "high"])
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn over_long_identifiers_are_rejected() {
        let long = "x".repeat(257);
        assert!(
            ticket()
                .noul("q", "?")
                .session_id(&long)
                .validate()
                .is_err()
        );
        assert!(ticket().noul("q", "?").user(&long).validate().is_err());
        assert!(
            ticket()
                .noul("q", "?")
                .session_id("fine")
                .validate()
                .is_ok()
        );
    }

    #[test]
    fn the_identifier_cap_counts_characters_not_bytes() {
        // 200 characters, 400 bytes: within the documented cap.
        let accented = "é".repeat(200);
        assert!(
            ticket()
                .noul("q", "?")
                .session_id(&accented)
                .validate()
                .is_ok()
        );
        assert!(ticket().noul("q", "?").user(&accented).validate().is_ok());
        assert!(
            ticket()
                .noul("q", "?")
                .session_id("é".repeat(257))
                .validate()
                .is_err()
        );
    }

    #[test]
    fn body_matches_the_documented_request_shape() {
        let body = ticket()
            .noul_described(
                "is_bug",
                "Is the customer reporting a software defect?",
                "The customer describes broken or unexpected product behavior.",
                "The customer is asking a question or requesting a feature.",
            )
            .choice(
                "team",
                "Which team should own this ticket?",
                [
                    ("payments", "Checkout, billing."),
                    ("frontend", "Rendering."),
                ],
            )
            .score("urgency", "How urgent?", ["next release", "this week"])
            .into_body("jev-1.13.0", Provider::TypeSafe);

        assert_eq!(body["model"], "jev-1.13.0");
        assert_eq!(body["state"]["ticket"], "blank screen after Pay");

        let noul = &body["questions"]["is_bug"];
        assert_eq!(noul["type"], "noul");
        assert_eq!(
            noul["instructions"],
            "Is the customer reporting a software defect?"
        );
        assert_eq!(
            noul["criteria"]["true"],
            "The customer describes broken or unexpected product behavior."
        );
        assert!(noul["criteria"]["false"].is_string());

        let choice = &body["questions"]["team"];
        assert_eq!(choice["type"], "choice");
        assert_eq!(choice["criteria"]["payments"], "Checkout, billing.");

        let score = &body["questions"]["urgency"];
        assert_eq!(score["type"], "score");
        assert_eq!(score["criteria"][0], "next release");
        assert_eq!(score["criteria"][1], "this week");
    }

    #[test]
    fn a_bare_noul_omits_criteria_entirely() {
        // The schema requires both sides when criteria is present, so an
        // undescribed noul must not emit a half-populated object.
        let body = ticket()
            .noul("is_bug", "Is this a defect?")
            .into_body("m", Provider::TypeSafe);
        assert!(body["questions"]["is_bug"].get("criteria").is_none());
    }

    #[test]
    fn a_bare_choice_sends_null_descriptions() {
        let body = ticket()
            .choice_bare("team", "Who?", ["payments", "frontend"])
            .into_body("m", Provider::TypeSafe);
        assert!(body["questions"]["team"]["criteria"]["payments"].is_null());
    }

    #[test]
    fn a_request_can_override_the_clients_model() {
        let body = ticket()
            .noul("q", "?")
            .model("typesafe/jev-9")
            .into_body("default", Provider::TypeSafe);
        assert_eq!(body["model"], "typesafe/jev-9");
    }

    #[test]
    fn a_plain_string_state_is_sent_as_a_string() {
        let body = Request::new("How many cards can I make per day?")
            .noul("q", "?")
            .into_body("m", Provider::TypeSafe);
        assert_eq!(body["state"], "How many cards can I make per day?");
    }

    #[test]
    fn optional_fields_are_omitted_when_unset() {
        let body = ticket().noul("q", "?").into_body("m", Provider::TypeSafe);
        assert!(body.get("session_id").is_none());
        assert!(body.get("user").is_none());
    }
}
