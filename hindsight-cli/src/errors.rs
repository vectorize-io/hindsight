use colored::*;

pub fn handle_api_error(err: anyhow::Error, api_url: &str) -> ! {
    eprintln!("{}", format_error_message(&err, api_url));
    std::process::exit(1);
}

/// Extract the server's own explanation from an error string produced by
/// `humanize_client_error` ("API request failed (404 Not Found): {body}").
///
/// Returns the JSON `detail` field when the body carries one, the raw body
/// otherwise, and `None` when there is no body at all. Every HTTP branch below
/// surfaces this instead of discarding it: a self-explanatory server response
/// beats generic guidance (see issues #2912, #4049).
fn server_detail(err_str: &str) -> Option<String> {
    // Errors carrying a body are shaped "<what> failed (<status>): <body>" —
    // `humanize_client_error` and the hand-rolled reqwest paths both use it.
    let (head, body) = err_str.split_once("): ")?;
    let status = head.rsplit_once('(')?.1;
    if !status.starts_with(|c: char| c.is_ascii_digit()) {
        return None;
    }
    let body = body.trim();
    if body.is_empty() {
        return None;
    }
    let Ok(json) = serde_json::from_str::<serde_json::Value>(body) else {
        return Some(body.to_string());
    };
    match json.get("detail") {
        Some(serde_json::Value::String(detail)) => Some(detail.clone()),
        Some(detail) => Some(detail.to_string()),
        None => Some(body.to_string()),
    }
}

/// A "Server response:" block for the detail, or nothing when the server sent
/// no body.
fn server_response_section(err_str: &str) -> String {
    match server_detail(err_str) {
        Some(detail) => format!(
            "\n\n{}\n  {}",
            "Server response:".bright_yellow(),
            detail.bright_white()
        ),
        None => String::new(),
    }
}

fn format_error_message(err: &anyhow::Error, api_url: &str) -> String {
    let err_str = err.to_string();

    // Connection refused
    if err_str.contains("Connection refused")
        || err_str.contains("tcp connect error")
        || err_str.contains("error sending request")
    {
        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n  • {}\n\n{}\n  {}",
            "✗".bright_red().bold(),
            "Cannot connect to Hindsight API".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "Possible causes:".bright_yellow(),
            "The Hindsight API server is not running".bright_white(),
            format!(
                "The server is running on a different address than {}",
                api_url
            )
            .bright_white(),
            "A firewall is blocking the connection".bright_white(),
            "Try:".bright_green(),
            "Start the Hindsight API server and ensure it's accessible".bright_white()
        );
    }

    // Preserve validation details before looking for timeout keywords. A fast
    // HTTP 400 may legitimately explain that a requested mode would time out;
    // classifying that body as a transport timeout hides the server's fix.
    if err_str.contains("400 Bad Request") || err_str.contains("(400)") {
        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  {}",
            "✗".bright_red().bold(),
            "Request rejected (400)".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "Server response:".bright_yellow(),
            server_detail(&err_str)
                .unwrap_or_else(|| err_str.clone())
                .bright_white()
        );
    }

    // Timeout
    if err_str.contains("timeout") || err_str.contains("Timeout") {
        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n\n{}\n  • {}\n  • {}",
            "✗".bright_red().bold(),
            "Request timed out".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "Possible causes:".bright_yellow(),
            "The API server is slow to respond".bright_white(),
            "Network latency is too high".bright_white(),
            "Try:".bright_green(),
            "Check if the API server is healthy".bright_white(),
            "Try again with a better network connection".bright_white()
        );
    }

    // DNS/Host resolution
    if err_str.contains("dns") || err_str.contains("DNS") || err_str.contains("failed to lookup") {
        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n\n{}\n  {}",
            "✗".bright_red().bold(),
            "Cannot resolve API hostname".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "Possible causes:".bright_yellow(),
            "The hostname in the API URL is incorrect".bright_white(),
            "DNS server is not responding".bright_white(),
            "Try:".bright_green(),
            "Check the HINDSIGHT_API_URL environment variable".bright_white()
        );
    }

    // 404 Not Found - check for disabled features first
    if err_str.contains("404") {
        if err_str.contains("Bank configuration API is disabled") {
            return format!(
                "{} {}\n\n{}\n  {}\n\n{}\n  {}\n\n{}\n  {}",
                "✗".bright_red().bold(),
                "Bank configuration API is disabled".bright_red().bold(),
                "API URL:".bright_yellow(),
                api_url.bright_white(),
                "This feature has been disabled on the server.".bright_yellow(),
                "To enable, set HINDSIGHT_API_ENABLE_BANK_CONFIG_API=true on the API server"
                    .bright_white(),
                "Note:".bright_cyan(),
                "This allows per-bank LLM configuration overrides via API".bright_white()
            );
        }

        // A 404 that explains itself ("Document not found") is about the
        // resource, not the route: print the server's words rather than
        // sending the operator after an API path/version mismatch.
        if let Some(detail) = server_detail(&err_str) {
            return format!(
                "{} {}\n\n{}\n  {}",
                "✗".bright_red().bold(),
                format!("Not found (404): {}", detail).bright_red().bold(),
                "API URL:".bright_yellow(),
                api_url.bright_white()
            );
        }

        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n\n{}\n  {}",
            "✗".bright_red().bold(),
            "API endpoint not found (404)".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "Possible causes:".bright_yellow(),
            "The API endpoint path has changed".bright_white(),
            "You're using an incompatible API version".bright_white(),
            "Try:".bright_green(),
            "Check that you're using the correct Hindsight API version".bright_white()
        );
    }

    // 401 Authentication failed
    if err_str.contains("401") {
        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n\n{}\n  {}{}",
            "✗".bright_red().bold(),
            "Authentication failed".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "Possible causes:".bright_yellow(),
            "API requires authentication".bright_white(),
            "Invalid or missing credentials".bright_white(),
            "Try:".bright_green(),
            "Check if the API requires an API key or token".bright_white(),
            server_response_section(&err_str)
        );
    }

    // 403 Forbidden
    if err_str.contains("403") {
        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n\n{}\n  {}{}",
            "✗".bright_red().bold(),
            "Permission denied (403)".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "Possible causes:".bright_yellow(),
            "This operation is not allowed".bright_white(),
            "The feature may be disabled on the server".bright_white(),
            "Try:".bright_green(),
            "Check server configuration or contact your administrator".bright_white(),
            server_response_section(&err_str)
        );
    }

    // 500 Server Error
    if err_str.contains("500") || err_str.contains("502") || err_str.contains("503") {
        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n\n{}\n  • {}\n  • {}{}",
            "✗".bright_red().bold(),
            "API server error".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "The server encountered an error:".bright_yellow(),
            "Internal server error (500)".bright_white(),
            "Service temporarily unavailable".bright_white(),
            "Try:".bright_green(),
            "Check the API server logs for details".bright_white(),
            "Try again in a few moments".bright_white(),
            server_response_section(&err_str)
        );
    }

    // Invalid URL
    if err_str.contains("invalid URL") || err_str.contains("InvalidUri") {
        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  {}\n\n{}\n  {}",
            "✗".bright_red().bold(),
            "Invalid API URL".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "The API URL format is invalid.".bright_yellow(),
            "Ensure it starts with http:// or https://".bright_white(),
            "Example:".bright_green(),
            "export HINDSIGHT_API_URL=http://localhost:8888".bright_white()
        );
    }

    // JSON parsing error - show actual response
    if err_str.contains("Failed to parse") || err_str.contains("error decoding") {
        // Extract the actual response if available
        let response_hint = if err_str.contains("Response was:") {
            let parts: Vec<&str> = err_str.split("Response was:").collect();
            if parts.len() > 1 {
                format!(
                    "\n{}\n{}",
                    "Actual response:".bright_yellow(),
                    parts[1].trim().bright_white()
                )
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n  • {}{}\n\n{}\n  • {}\n  • {}",
            "✗".bright_red().bold(),
            "Invalid API response format".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "Possible causes:".bright_yellow(),
            "The API returned an unexpected response format".bright_white(),
            "Version mismatch between CLI and API".bright_white(),
            "The API endpoint doesn't exist or returned HTML instead of JSON".bright_white(),
            response_hint,
            "Try:".bright_green(),
            "Run with --verbose flag to see the full request/response".bright_white(),
            "Ensure you're using a compatible Hindsight API version".bright_white()
        );
    }

    // Generic error with the full error message
    format!(
        "{} {}\n\n{}\n  {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n  • {}",
        "✗".bright_red().bold(),
        "API request failed".bright_red().bold(),
        "API URL:".bright_yellow(),
        api_url.bright_white(),
        "Error:".bright_yellow(),
        err_str.bright_white(),
        "Suggestions:".bright_green(),
        "Check that HINDSIGHT_API_URL is set correctly".bright_white(),
        "Ensure the Hindsight API server is running".bright_white(),
        "Verify network connectivity to the API server".bright_white()
    )
}

pub fn print_config_help() {
    println!("\n{}", "Configuration:".bright_cyan().bold());
    println!("  Run the configure command to set the API URL:");
    println!("  {}", "hindsight configure".bright_white());
    println!();
    println!("  Or set it directly:");
    println!(
        "  {}",
        "hindsight configure --api-url http://your-api:8888".bright_white()
    );
    println!();
    println!("  {}", "Configuration priority:".bright_yellow());
    println!("    1. Environment variable (HINDSIGHT_API_URL) - highest priority");
    println!("    2. Config file (~/.hindsight/config)");
    println!("    3. Default (http://localhost:8888)");
    println!();
}

#[cfg(test)]
mod tests {
    use super::format_error_message;

    #[test]
    fn http_400_body_that_mentions_timeout_is_not_reported_as_a_timeout() {
        let error = anyhow::anyhow!(
            "API request failed (400 Bad Request): \
             {{\"detail\":\"Batch operations will timeout in synchronous mode. Please set async=true.\"}}"
        );

        let message = format_error_message(&error, "http://localhost:8888");

        assert!(message.contains("Batch operations will timeout in synchronous mode"));
        assert!(!message.contains("Request timed out"));
    }

    #[test]
    fn http_404_with_a_detail_reports_the_server_message() {
        let error = anyhow::anyhow!(
            "API request failed (404 Not Found): {{\"detail\":\"Document not found\"}}"
        );

        let message = format_error_message(&error, "http://localhost:8888");

        assert!(message.contains("Not found (404): Document not found"));
        assert!(!message.contains("API endpoint not found"));
        assert!(!message.contains("incompatible API version"));
    }

    #[test]
    fn http_404_without_a_body_keeps_the_unknown_route_guidance() {
        let error = anyhow::anyhow!("API request failed (404 Not Found)");

        let message = format_error_message(&error, "http://localhost:8888");

        assert!(message.contains("API endpoint not found (404)"));
        assert!(message.contains("incompatible API version"));
    }

    #[test]
    fn http_404_for_the_disabled_bank_config_api_keeps_its_dedicated_help() {
        let error = anyhow::anyhow!(
            "API request failed (404 Not Found): \
             {{\"detail\":\"Bank configuration API is disabled\"}}"
        );

        let message = format_error_message(&error, "http://localhost:8888");

        assert!(message.contains("HINDSIGHT_API_ENABLE_BANK_CONFIG_API=true"));
    }

    #[test]
    fn http_403_surfaces_the_server_response() {
        let error = anyhow::anyhow!(
            "API request failed (403 Forbidden): {{\"detail\":\"Bank is read-only\"}}"
        );

        let message = format_error_message(&error, "http://localhost:8888");

        assert!(message.contains("Permission denied (403)"));
        assert!(message.contains("Bank is read-only"));
    }

    #[test]
    fn http_500_surfaces_the_server_response() {
        let error = anyhow::anyhow!(
            "API request failed (500 Internal Server Error): {{\"detail\":\"embedding backend unreachable\"}}"
        );

        let message = format_error_message(&error, "http://localhost:8888");

        assert!(message.contains("embedding backend unreachable"));
    }

    #[test]
    fn a_body_from_the_hand_rolled_reqwest_paths_is_surfaced_too() {
        let error = anyhow::anyhow!("Import failed (404 Not Found): bank does not exist");

        let message = format_error_message(&error, "http://localhost:8888");

        assert!(message.contains("Not found (404): bank does not exist"));
    }

    #[test]
    fn a_parenthetical_that_is_not_a_status_is_not_read_as_a_body() {
        let error = anyhow::anyhow!("some failure (not a status): 404 somewhere in the text");

        let message = format_error_message(&error, "http://localhost:8888");

        assert!(message.contains("API endpoint not found (404)"));
    }

    #[test]
    fn a_non_json_body_is_shown_verbatim() {
        let error = anyhow::anyhow!("API request failed (404 Not Found): <html>nginx 404</html>");

        let message = format_error_message(&error, "http://localhost:8888");

        assert!(message.contains("<html>nginx 404</html>"));
    }
}
