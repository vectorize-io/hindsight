from urllib.parse import urlparse, urlunparse

def mask_network_location(url):
    if not url:
        return url
    parsed_url = urlparse(url)
    masked_network_location = parsed_url.hostname or ""
    if parsed_url.port:
        masked_network_location += f":{parsed_url.port}"
    if parsed_url.username or parsed_url.password:
        masked_network_location = f"***:***@{masked_network_location}"
    return urlunparse(parsed_url._replace(netloc=masked_network_location))