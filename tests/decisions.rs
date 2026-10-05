//! The client driven end to end against a stub decision endpoint.
//!
//! The stub is a plain `TcpListener` speaking just enough HTTP/1.1, so the
//! real request is built, serialised, sent, and the real response parsed —
//! without a network call, an API key, or an HTTP-server dependency.

mod common;

use arbiter::{Arbiter, Error, Provider, Request};
use common::DOCUMENTED;
use serde_json::json;
use tokio::sync::oneshot;

/// A client on OpenRouter pointed at a stub that answers once with `body`.
async fn stub_client(status: u16, body: &str) -> (Arbiter, oneshot::Receiver<common::Received>) {
    common::stub_client(Provider::OpenRouter, "sk-or-v1-test-key", status, body).await
}

fn ticket() -> Request {
    Request::new(json!({"ticket": "blank screen after Pay", "customer_tier": "enterprise"}))
        .noul_described(
            "is_bug",
            "Is the customer reporting a software defect?",
            "Describes broken behavior.",
            "Asks a question.",
        )
        .choice(
            "team",
            "Which team should own this ticket?",
            [
                ("payments", "Checkout, billing."),
                ("frontend", "Rendering."),
            ],
        )
        .score(
            "urgency",
            "How urgent?",
            ["Can wait", "This week", "Blocking now"],
        )
}

#[tokio::test]
async fn answers_a_batch_of_three_primitives_in_one_call() {
    let (client, received) = stub_client(200, DOCUMENTED).await;

    let answers = client.ask(ticket()).await.expect("the stub answers");

    assert_eq!(answers.len(), 3, "one round trip answered every question");
    assert!((answers.noul("is_bug").expect("noul").probability - 0.96).abs() < 1e-9);
    assert_eq!(answers.choice("team").expect("choice").choice, "payments");
    assert_eq!(answers.score("urgency").expect("score").nearest_level(), 2);
    assert_eq!(answers.model, "typesafe/jev-1.13-20260917");
    assert_eq!(answers.usage.cost, Some(0.000_019_992));

    let sent = received.await.expect("the stub recorded the request");
    assert_eq!(sent.path, "/api/alpha/decisions");
}

#[tokio::test]
async fn sends_the_key_as_a_bearer_token_and_never_in_the_body() {
    let (client, received) = stub_client(200, DOCUMENTED).await;
    client.ask(ticket()).await.expect("the stub answers");

    let sent = received.await.expect("recorded");
    assert_eq!(
        sent.authorization.as_deref(),
        Some("Bearer sk-or-v1-test-key")
    );
    assert!(
        !sent.body.to_string().contains("sk-or-v1-test-key"),
        "the credential must travel in the header, not the payload"
    );
}

#[tokio::test]
async fn serialises_every_primitive_in_its_documented_shape() {
    let (client, received) = stub_client(200, DOCUMENTED).await;
    client.ask(ticket()).await.expect("the stub answers");
    let body = received.await.expect("recorded").body;

    assert_eq!(body["model"], "typesafe/jev-1.13");
    assert_eq!(body["state"]["customer_tier"], "enterprise");

    assert_eq!(body["questions"]["is_bug"]["type"], "noul");
    assert_eq!(
        body["questions"]["is_bug"]["criteria"]["true"],
        "Describes broken behavior."
    );

    assert_eq!(body["questions"]["team"]["type"], "choice");
    assert_eq!(
        body["questions"]["team"]["criteria"]["payments"],
        "Checkout, billing."
    );

    assert_eq!(body["questions"]["urgency"]["type"], "score");
    assert_eq!(body["questions"]["urgency"]["criteria"][2], "Blocking now");
}

#[tokio::test]
async fn a_request_level_model_overrides_the_clients() {
    let (client, received) = stub_client(200, DOCUMENTED).await;
    client
        .ask(ticket().model("~typesafe/jev-latest"))
        .await
        .expect("the stub answers");

    let body = received.await.expect("recorded").body;
    assert_eq!(body["model"], "~typesafe/jev-latest");
}

#[tokio::test]
async fn surfaces_the_providers_own_error_message() {
    let error_body = r#"{"error":{"code":402,"message":"Insufficient credits. Add more using https://openrouter.ai/credits"}}"#;
    let (client, _received) = stub_client(402, error_body).await;

    match client.ask(ticket()).await {
        Err(Error::Api { status, message }) => {
            assert_eq!(status, 402);
            assert!(message.contains("Insufficient credits"), "{message}");
        }
        other => panic!("expected Api, got {other:?}"),
    }
}

#[tokio::test]
async fn classifies_an_overloaded_provider_as_transient() {
    let (client, _received) = stub_client(
        529,
        r#"{"error":{"code":529,"message":"Provider returned error"}}"#,
    )
    .await;

    let error = client.ask(ticket()).await.expect_err("529 must fail");
    assert!(error.is_transient(), "529 is worth retrying: {error}");
}

#[tokio::test]
async fn classifies_a_bad_request_as_permanent() {
    let (client, _received) = stub_client(
        400,
        r#"{"error":{"code":400,"message":"Invalid request parameters"}}"#,
    )
    .await;

    let error = client.ask(ticket()).await.expect_err("400 must fail");
    assert!(!error.is_transient(), "400 will fail identically on retry");
}

#[tokio::test]
async fn reports_the_raw_body_when_the_response_is_not_the_documented_shape() {
    let (client, _received) = stub_client(200, r#"{"unexpected":"shape"}"#).await;

    match client.ask(ticket()).await {
        Err(Error::Decode { raw, .. }) => assert!(raw.contains("unexpected"), "{raw}"),
        other => panic!("expected Decode, got {other:?}"),
    }
}

#[tokio::test]
async fn validates_before_spending_a_request() {
    // Nothing is listening on this port: reaching the network at all would
    // surface as a transport error instead of a request error.
    let client = Arbiter::new("k")
        .expect("build")
        .base_url("http://127.0.0.1:1");

    let underspecified = Request::new("state").choice_bare("only", "Which?", ["just-one"]);
    assert!(matches!(
        client.ask(underspecified).await,
        Err(Error::Request(_))
    ));

    assert!(matches!(
        client.ask(Request::new("state")).await,
        Err(Error::Request(_))
    ));
}

#[tokio::test]
async fn a_raw_body_is_sent_verbatim() {
    let (client, received) = stub_client(200, DOCUMENTED).await;

    client
        .ask_raw(json!({
            "model": "typesafe/jev-1.13",
            "state": "anything",
            "questions": {"q": {"type": "noul", "instructions": "?"}},
            "session_id": "batch-7"
        }))
        .await
        .expect("the stub answers");

    let body = received.await.expect("recorded").body;
    assert_eq!(body["session_id"], "batch-7");
    assert_eq!(body["state"], "anything");
}

#[tokio::test]
async fn a_recorded_exchange_keeps_both_halves_verbatim() {
    // The point of `ask_recorded`: a caller showing what crossed the wire
    // must get the bytes, not a re-serialisation of the parsed answers.
    let (client, received) = stub_client(200, DOCUMENTED).await;

    let exchange = client
        .ask_recorded(ticket())
        .await
        .expect("the stub answers");

    // The request half is exactly what was posted.
    let sent = received.await.expect("recorded").body;
    assert_eq!(exchange.request, sent);

    // The response half is the body as it arrived, not a round trip.
    assert_eq!(exchange.response, DOCUMENTED);
    assert_eq!(exchange.answers.len(), 3);
    assert_eq!(
        exchange.answers.choice("team").expect("choice").choice,
        "payments"
    );
}

#[tokio::test]
async fn a_recorded_exchange_validates_before_sending() {
    // Nothing is listening on this port: reaching the network at all would
    // surface as a transport error instead of a request error.
    let client = Arbiter::new("k")
        .expect("build")
        .base_url("http://127.0.0.1:1");

    assert!(matches!(
        client.ask_recorded(Request::new("state")).await,
        Err(Error::Request(_))
    ));
}

/// A TypeSafe response: the documented subset, with no id, provider, or cost.
const TYPESAFE_RESPONSE: &str = r#"{
  "model": "jev-1.13.0",
  "answers": { "is_bug": { "type": "noul", "noul": 0.95 } },
  "usage": { "input_tokens": 296, "output_tokens": 20 }
}"#;

/// A client on TypeSafe pointed at a stub.
async fn typesafe_client(body: &str) -> (Arbiter, oneshot::Receiver<common::Received>) {
    common::stub_client(Provider::TypeSafe, "ts-test-key", 200, body).await
}

#[tokio::test]
async fn typesafe_posts_to_its_own_path_with_its_own_model() {
    let (client, received) = typesafe_client(TYPESAFE_RESPONSE).await;
    assert_eq!(client.provider(), arbiter::Provider::TypeSafe);

    let answers = client
        .ask(Request::new("payouts failing").noul("is_bug", "A defect?"))
        .await
        .expect("the stub answers");

    assert!((answers.noul("is_bug").expect("noul").probability - 0.95).abs() < 1e-9);
    assert_eq!(answers.model, "jev-1.13.0");

    let sent = received.await.expect("recorded");
    assert_eq!(sent.path, "/v1/systemone", "TypeSafe has its own path");
    assert_eq!(sent.authorization.as_deref(), Some("Bearer ts-test-key"));
    assert_eq!(
        sent.body["model"], "jev-1.13.0",
        "and its own model vocabulary"
    );
}

#[tokio::test]
async fn a_typesafe_response_parses_without_the_openrouter_only_fields() {
    // TypeSafe omits id, provider and usage.cost; all three are optional.
    let (client, _received) = typesafe_client(TYPESAFE_RESPONSE).await;

    let answers = client
        .ask(Request::new("x").noul("is_bug", "A defect?"))
        .await
        .expect("the stub answers");

    assert!(answers.id.is_none());
    assert!(answers.provider.is_none());
    assert!(answers.usage.cost.is_none());
    assert_eq!(answers.usage.input_tokens, 296);
}

#[tokio::test]
async fn attribution_fields_are_withheld_from_typesafe_but_sent_to_openrouter() {
    // TypeSafe documents the body as state/model/questions only, so sending
    // OpenRouter's extensions there risks a validation failure.
    let (client, received) = typesafe_client(TYPESAFE_RESPONSE).await;
    client
        .ask(
            Request::new("x")
                .noul("is_bug", "A defect?")
                .session_id("s-1")
                .user("u-1"),
        )
        .await
        .expect("the stub answers");

    let sent = received.await.expect("recorded").body;
    assert!(sent.get("session_id").is_none(), "{sent}");
    assert!(sent.get("user").is_none(), "{sent}");

    let (client, received) = stub_client(200, DOCUMENTED).await;
    client
        .ask(
            Request::new("x")
                .noul("is_bug", "A defect?")
                .session_id("s-1")
                .user("u-1"),
        )
        .await
        .expect("the stub answers");

    let sent = received.await.expect("recorded").body;
    assert_eq!(sent["session_id"], "s-1");
    assert_eq!(sent["user"], "u-1");
}

#[tokio::test]
async fn typesafes_validation_status_is_not_treated_as_retryable() {
    // TypeSafe answers 422 where OpenRouter answers 400; neither improves on
    // a retry.
    let stub = common::stub(422, r#"{"error":{"message":"missing field"}}"#).await;
    let client = Arbiter::new("k").expect("build").base_url(stub.url);

    let error = client
        .ask(Request::new("x").noul("q", "?"))
        .await
        .expect_err("422 must fail");

    assert!(!error.is_transient(), "422 is permanent: {error}");
    match error {
        Error::Api { status, .. } => assert_eq!(status, 422),
        other => panic!("expected Api, got {other:?}"),
    }
}
