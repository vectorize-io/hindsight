use colored::*;

pub fn handle_api_error(err: anyhow::Error, api_url: &str) -> ! {
    eprintln!("{}", format_error_message(&err, api_url));
    std::process::exit(1);
}

fn format_error_message(err: &anyhow::Error, api_url: &str) -> String {
    let err_str = err.to_string();

    // Connection refused
    if err_str.contains("Connection refused") || err_str.contains("tcp connect error") || err_str.contains("error sending request") {
        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n  • {}\n\n{}\n  {}",
            "✗".bright_red().bold(),
            "Cannot connect to Hindsight API".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "Possible causes:".bright_yellow(),
            "The Hindsight API server is not running".bright_white(),
            format!("The server is running on a different address than {}", api_url).bright_white(),
            "A firewall is blocking the connection".bright_white(),
            "Try:".bright_green(),
            "Start the Hindsight API server and ensure it's accessible".bright_white()
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
                "This feature is disabled by default for security.".bright_yellow(),
                "To enable, set HINDSIGHT_API_ENABLE_BANK_CONFIG_API=true on the API server".bright_white(),
                "Note:".bright_cyan(),
                "This allows per-bank LLM configuration overrides via API".bright_white()
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
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n\n{}\n  {}",
            "✗".bright_red().bold(),
            "Authentication failed".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "Possible causes:".bright_yellow(),
            "API requires authentication".bright_white(),
            "Invalid or missing credentials".bright_white(),
            "Try:".bright_green(),
            "Check if the API requires an API key or token".bright_white()
        );
    }

    // 403 Forbidden
    if err_str.contains("403") {
        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n\n{}\n  {}",
            "✗".bright_red().bold(),
            "Permission denied (403)".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "Possible causes:".bright_yellow(),
            "This operation is not allowed".bright_white(),
            "The feature may be disabled on the server".bright_white(),
            "Try:".bright_green(),
            "Check server configuration or contact your administrator".bright_white()
        );
    }

    // 500 Server Error
    if err_str.contains("500") || err_str.contains("502") || err_str.contains("503") {
        return format!(
            "{} {}\n\n{}\n  {}\n\n{}\n  • {}\n  • {}\n\n{}\n  • {}\n  • {}",
            "✗".bright_red().bold(),
            "API server error".bright_red().bold(),
            "API URL:".bright_yellow(),
            api_url.bright_white(),
            "The server encountered an error:".bright_yellow(),
            "Internal server error (500)".bright_white(),
            "Service temporarily unavailable".bright_white(),
            "Try:".bright_green(),
            "Check the API server logs for details".bright_white(),
            "Try again in a few moments".bright_white()
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
                format!("\n{}\n{}", "Actual response:".bright_yellow(), parts[1].trim().bright_white())
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
    println!("  {}", "hindsight configure --api-url http://your-api:8888".bright_white());
    println!();
    println!("  {}", "Configuration priority:".bright_yellow());
    println!("    1. Environment variable (HINDSIGHT_API_URL) - highest priority");
    println!("    2. Config file (~/.hindsight/config)");
    println!("    3. Default (http://localhost:8888)");
    println!();
}
