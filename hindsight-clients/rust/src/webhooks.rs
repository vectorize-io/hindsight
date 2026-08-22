//! Consumer helpers for authenticating and parsing Hindsight webhooks.

use hmac::{Hmac, Mac};
use http::HeaderMap;
use serde::Deserialize;
use sha2::Sha256;

use crate::types;

pub const SIGNATURE_HEADER: &str = "X-Hindsight-Signature";

#[derive(Clone, Debug)]
pub enum ConsumedWebhookEvent {
    Known(Box<types::WebhookEvent>),
    Unknown(types::WebhookEventEnvelope),
}

impl ConsumedWebhookEvent {
    pub fn event_name(&self) -> &str {
        match self {
            Self::Known(event) => match event.as_ref() {
                types::WebhookEvent::ConsolidationCompletedWebhookEvent(_) => {
                    "consolidation.completed"
                }
                types::WebhookEvent::RetainCompletedWebhookEvent(_) => "retain.completed",
                types::WebhookEvent::MemoryDefenseTriggeredWebhookEvent(_) => {
                    "memory_defense.triggered"
                }
            },
            Self::Unknown(event) => &event.event,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WebhookError {
    #[error("webhook signature verification failed")]
    InvalidSignature,
    #[error("invalid webhook payload: {0}")]
    InvalidPayload(String),
}

fn decode_signature(signature: &str) -> Option<[u8; 32]> {
    let digest = signature.strip_prefix("sha256=")?;
    if digest.len() != 64 {
        return None;
    }

    let mut result = [0_u8; 32];
    for (index, pair) in digest.as_bytes().chunks_exact(2).enumerate() {
        let high = decode_hex_digit(pair[0])?;
        let low = decode_hex_digit(pair[1])?;
        result[index] = high << 4 | low;
    }
    Some(result)
}

fn decode_hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

/// Verify the current `sha256=<hex>` signature over the exact raw body.
pub fn verify_signature(raw_body: &[u8], signature: &str, secret: &str) -> bool {
    let Some(provided) = decode_signature(signature) else {
        return false;
    };

    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())
        .expect("HMAC accepts keys of every length");
    mac.update(raw_body);
    mac.verify_slice(&provided).is_ok()
}

#[derive(Deserialize)]
struct EventDiscriminator {
    event: String,
}

/// Parse an event without authenticating it.
///
/// Prefer [`construct_event`] unless the webhook is explicitly unsigned.
pub fn parse_event(raw_body: &[u8]) -> Result<ConsumedWebhookEvent, WebhookError> {
    let discriminator: EventDiscriminator = serde_json::from_slice(raw_body)
        .map_err(|error| WebhookError::InvalidPayload(error.to_string()))?;

    match discriminator.event.as_str() {
        "consolidation.completed" | "retain.completed" | "memory_defense.triggered" => {
            serde_json::from_slice(raw_body)
                .map(Box::new)
                .map(ConsumedWebhookEvent::Known)
                .map_err(|error| WebhookError::InvalidPayload(error.to_string()))
        }
        _ => serde_json::from_slice(raw_body)
            .map(ConsumedWebhookEvent::Unknown)
            .map_err(|error| WebhookError::InvalidPayload(error.to_string())),
    }
}

/// Verify a signed raw request body before parsing it.
///
/// `X-Hindsight-Event` is intentionally ignored because it is not signed.
pub fn construct_event(
    raw_body: &[u8],
    headers: &HeaderMap,
    secret: &str,
) -> Result<ConsumedWebhookEvent, WebhookError> {
    let signature = headers
        .get(SIGNATURE_HEADER)
        .and_then(|value| value.to_str().ok())
        .ok_or(WebhookError::InvalidSignature)?;

    if !verify_signature(raw_body, signature, secret) {
        return Err(WebhookError::InvalidSignature);
    }
    parse_event(raw_body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Deserialize)]
    struct SignatureVector {
        secret: String,
        raw_body: String,
        signature: String,
    }

    fn vector() -> SignatureVector {
        serde_json::from_str(include_str!("../../testdata/webhook-signatures.json")).unwrap()
    }

    fn sign(raw_body: &[u8], secret: &str) -> String {
        let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes()).unwrap();
        mac.update(raw_body);
        let digest = mac.finalize().into_bytes();
        let hex = digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        format!("sha256={hex}")
    }

    #[test]
    fn verifies_shared_vector_over_exact_bytes() {
        let vector = vector();
        assert!(verify_signature(
            vector.raw_body.as_bytes(),
            &vector.signature,
            &vector.secret
        ));
        assert!(!verify_signature(
            format!("{}\n", vector.raw_body).as_bytes(),
            &vector.signature,
            &vector.secret
        ));
        assert!(!verify_signature(
            vector.raw_body.as_bytes(),
            &vector.signature.to_uppercase(),
            &vector.secret
        ));
    }

    #[test]
    fn constructs_generated_known_event() {
        let vector = vector();
        let mut headers = HeaderMap::new();
        headers.insert(SIGNATURE_HEADER, vector.signature.parse().unwrap());

        let event = construct_event(vector.raw_body.as_bytes(), &headers, &vector.secret).unwrap();
        match event {
            ConsumedWebhookEvent::Known(event) => match *event {
                types::WebhookEvent::RetainCompletedWebhookEvent(event) => {
                    assert_eq!(event.data.document_id.as_deref(), Some("doc-1"));
                }
                other => panic!("unexpected event: {other:?}"),
            },
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn verifies_before_parsing_json() {
        let raw_body = b"{";
        let mut headers = HeaderMap::new();
        headers.insert(
            SIGNATURE_HEADER,
            format!("sha256={}", "0".repeat(64)).parse().unwrap(),
        );
        assert!(matches!(
            construct_event(raw_body, &headers, "secret"),
            Err(WebhookError::InvalidSignature)
        ));

        headers.insert(SIGNATURE_HEADER, sign(raw_body, "secret").parse().unwrap());
        assert!(matches!(
            construct_event(raw_body, &headers, "secret"),
            Err(WebhookError::InvalidPayload(_))
        ));
    }

    #[test]
    fn unknown_event_uses_forward_compatible_envelope() {
        let raw_body = br#"{"event":"future.created","bank_id":"bank-1","operation_id":"op-2","status":"completed","timestamp":"2026-08-11T09:30:00Z","data":{"future":true},"future_envelope_field":42}"#;

        let event = parse_event(raw_body).unwrap();
        match event {
            ConsumedWebhookEvent::Unknown(event) => {
                assert_eq!(event.event, "future.created");
                assert_eq!(event.data["future"], true);
                assert_eq!(event.extra["future_envelope_field"], 42);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    #[test]
    fn generated_discriminators_select_each_known_event() {
        let event = parse_event(br#"{"event":"memory_defense.triggered","bank_id":"bank-1","operation_id":"op-3","status":"block","timestamp":"2026-08-11T09:30:00Z","data":{"action":"block","future_data_field":true},"future_envelope_field":42}"#).unwrap();

        match event {
            ConsumedWebhookEvent::Known(event) => match *event {
                types::WebhookEvent::MemoryDefenseTriggeredWebhookEvent(event) => {
                    assert_eq!(event.extra["future_envelope_field"], 42);
                    assert_eq!(event.data.extra["future_data_field"], true);
                }
                other => panic!("unexpected event: {other:?}"),
            },
            other => panic!("unexpected event: {other:?}"),
        }
    }
}
