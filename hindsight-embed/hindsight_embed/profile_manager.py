"""Profile management for hindsight-embed.

Handles creation, deletion, and management of configuration profiles.
Each profile has its own config, daemon lock, log file, and port.
"""

import fcntl
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# Configuration paths
CONFIG_DIR = Path.home() / ".hindsight"
PROFILES_DIR = CONFIG_DIR / "profiles"
METADATA_FILE = PROFILES_DIR / "metadata.json"
ACTIVE_PROFILE_FILE = CONFIG_DIR / "active_profile"

# Port allocation
DEFAULT_PORT = 8888
PROFILE_PORT_BASE = 8889
PROFILE_PORT_RANGE = 1000  # 8889-9888


@dataclass
class ProfilePaths:
    """Paths and port for a profile."""

    config: Path
    lock: Path
    log: Path
    port: int


@dataclass
class ProfileInfo:
    """Profile information including metadata."""

    name: str
    port: int
    created_at: str
    last_used: Optional[str] = None
    is_active: bool = False
    daemon_running: bool = False


@dataclass
class ProfileMetadata:
    """Metadata for all profiles."""

    version: int = 1
    profiles: dict[str, dict] = field(default_factory=dict)


class ProfileManager:
    """Manages configuration profiles for hindsight-embed."""

    def __init__(self):
        """Initialize the profile manager."""
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure profile directories exist."""
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)

    def list_profiles(self) -> list[ProfileInfo]:
        """List all profiles with their status.

        Returns:
            List of ProfileInfo objects with daemon status.
        """
        metadata = self._load_metadata()
        active_profile = self.get_active_profile()
        profiles = []

        # Add default profile if config exists
        default_config = CONFIG_DIR / "embed"
        if default_config.exists():
            profiles.append(
                ProfileInfo(
                    name="",  # Empty name = default
                    port=DEFAULT_PORT,
                    created_at="",  # Don't track for default
                    last_used=None,
                    is_active=active_profile == "",
                    daemon_running=self._check_daemon_running(DEFAULT_PORT),
                )
            )

        # Add named profiles
        for name, info in metadata.profiles.items():
            profiles.append(
                ProfileInfo(
                    name=name,
                    port=info["port"],
                    created_at=info.get("created_at", ""),
                    last_used=info.get("last_used"),
                    is_active=active_profile == name,
                    daemon_running=self._check_daemon_running(info["port"]),
                )
            )

        return sorted(profiles, key=lambda p: (p.name != "", p.name))

    def profile_exists(self, name: str) -> bool:
        """Check if a profile exists.

        Args:
            name: Profile name (empty string for default).

        Returns:
            True if profile exists.
        """
        if not name:
            # Default profile exists if config file exists
            return (CONFIG_DIR / "embed").exists()

        # Named profile exists if config file exists
        config_path = PROFILES_DIR / f"{name}.env"
        return config_path.exists()

    def get_profile(self, name: str) -> Optional[ProfileInfo]:
        """Get profile information.

        Args:
            name: Profile name (empty string for default).

        Returns:
            ProfileInfo if profile exists, None otherwise.
        """
        profiles = self.list_profiles()
        for profile in profiles:
            if profile.name == name:
                return profile
        return None

    def create_profile(self, name: str, port_or_config: int | dict[str, str], config: dict[str, str] | None = None):
        """Create or update a profile.

        Args:
            name: Profile name.
            port_or_config: Port number (int) or configuration dict. For backward compatibility,
                            if this is a dict, it's treated as config and port is auto-allocated.
            config: Configuration dict (KEY=VALUE pairs). Only used if port_or_config is an int.

        Raises:
            ValueError: If profile name is invalid or port is invalid.
        """
        # Handle backward compatibility - allow (name, config) or (name, port, config)
        if isinstance(port_or_config, dict):
            # Called with (name, config) - auto-allocate port
            port = None
            config = port_or_config
        else:
            # Called with (name, port, config)
            port = port_or_config
            if config is None:
                raise ValueError("Config must be provided when port is specified")

        if not name:
            raise ValueError("Profile name cannot be empty")

        if not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid profile name '{name}'. Use alphanumeric chars, hyphens, and underscores.")

        if port is not None and (port < 1024 or port > 65535):
            raise ValueError(f"Invalid port {port}. Must be between 1024-65535.")

        # Ensure profile directory exists
        self._ensure_directories()

        # Load metadata to check if profile already exists
        metadata = self._load_metadata()

        # Determine port: use provided port, preserve existing, or allocate new
        if port is None:
            if name in metadata.profiles and "port" in metadata.profiles[name]:
                port = metadata.profiles[name]["port"]
            else:
                port = self._allocate_port(name)

        # Write config file
        config_path = PROFILES_DIR / f"{name}.env"
        config_lines = [f"{key}={value}" for key, value in config.items()]
        config_path.write_text("\n".join(config_lines) + "\n")

        # Update metadata (reuse metadata loaded earlier to avoid race conditions)
        now_iso = datetime.now(timezone.utc).isoformat()

        if name in metadata.profiles:
            # Update existing profile
            metadata.profiles[name]["last_used"] = now_iso
            metadata.profiles[name]["port"] = port
        else:
            # Create new profile
            metadata.profiles[name] = {
                "port": port,
                "created_at": now_iso,
                "last_used": now_iso,
            }

        self._save_metadata(metadata)

    def delete_profile(self, name: str):
        """Delete a profile.

        Args:
            name: Profile name.

        Raises:
            ValueError: If profile name is invalid or doesn't exist.
        """
        if not name:
            raise ValueError("Cannot delete default profile")

        if not self.profile_exists(name):
            raise ValueError(f"Profile '{name}' does not exist")

        # Remove config file
        config_path = PROFILES_DIR / f"{name}.env"
        if config_path.exists():
            config_path.unlink()

        # Remove lock file
        lock_path = PROFILES_DIR / f"{name}.lock"
        if lock_path.exists():
            lock_path.unlink()

        # Remove log file
        log_path = PROFILES_DIR / f"{name}.log"
        if log_path.exists():
            log_path.unlink()

        # Update metadata
        metadata = self._load_metadata()
        if name in metadata.profiles:
            del metadata.profiles[name]
            self._save_metadata(metadata)

        # Clear active profile if it was deleted
        if self.get_active_profile() == name:
            self.set_active_profile(None)

    def set_active_profile(self, name: Optional[str]):
        """Set the active profile.

        Args:
            name: Profile name to activate, or None to clear.

        Raises:
            ValueError: If profile doesn't exist.
        """
        if name and not self.profile_exists(name):
            raise ValueError(f"Profile '{name}' does not exist")

        if name:
            ACTIVE_PROFILE_FILE.write_text(name)
        else:
            # Clear active profile
            if ACTIVE_PROFILE_FILE.exists():
                ACTIVE_PROFILE_FILE.unlink()

    def get_active_profile(self) -> str:
        """Get the currently active profile name.

        Returns:
            Profile name, or empty string if no active profile.
        """
        if ACTIVE_PROFILE_FILE.exists():
            return ACTIVE_PROFILE_FILE.read_text().strip()
        return ""

    def resolve_profile_paths(self, name: str) -> ProfilePaths:
        """Resolve paths for a profile.

        Args:
            name: Profile name (empty string for default).

        Returns:
            ProfilePaths with config, lock, log, and port.
        """
        if not name:
            # Default profile
            return ProfilePaths(
                config=CONFIG_DIR / "embed",
                lock=CONFIG_DIR / "daemon.lock",
                log=CONFIG_DIR / "daemon.log",
                port=DEFAULT_PORT,
            )

        # Named profile
        metadata = self._load_metadata()
        port = metadata.profiles.get(name, {}).get("port", self._allocate_port(name))

        return ProfilePaths(
            config=PROFILES_DIR / f"{name}.env",
            lock=PROFILES_DIR / f"{name}.lock",
            log=PROFILES_DIR / f"{name}.log",
            port=port,
        )

    def _allocate_port(self, name: str) -> int:
        """Allocate a port for a profile using hash-based strategy.

        Args:
            name: Profile name.

        Returns:
            Port number (8889-9888).
        """
        # Hash profile name to get consistent port
        hash_val = int(hashlib.sha256(name.encode()).hexdigest(), 16)
        port = PROFILE_PORT_BASE + (hash_val % PROFILE_PORT_RANGE)

        # Check if port is already allocated in metadata
        metadata = self._load_metadata()
        allocated_ports = {info["port"] for info in metadata.profiles.values() if info.get("port")}

        # If collision, find next available port
        attempt = 0
        while port in allocated_ports and attempt < PROFILE_PORT_RANGE:
            port = PROFILE_PORT_BASE + ((hash_val + attempt) % PROFILE_PORT_RANGE)
            attempt += 1

        if attempt >= PROFILE_PORT_RANGE:
            # Fallback: find first available port
            for p in range(PROFILE_PORT_BASE, PROFILE_PORT_BASE + PROFILE_PORT_RANGE):
                if p not in allocated_ports:
                    return p
            raise RuntimeError("No available ports for profile")

        return port

    def _check_daemon_running(self, port: int) -> bool:
        """Check if daemon is running on a port.

        Args:
            port: Port number to check.

        Returns:
            True if daemon is responding.
        """
        try:
            with httpx.Client() as client:
                response = client.get(f"http://127.0.0.1:{port}/health", timeout=1)
                return response.status_code == 200
        except Exception:
            return False

    def _load_metadata(self) -> ProfileMetadata:
        """Load profile metadata from disk.

        Returns:
            ProfileMetadata object.
        """
        if not METADATA_FILE.exists():
            return ProfileMetadata()

        try:
            with open(METADATA_FILE) as f:
                data = json.load(f)
                return ProfileMetadata(version=data.get("version", 1), profiles=data.get("profiles", {}))
        except (json.JSONDecodeError, IOError) as e:
            print(
                f"Warning: Failed to load metadata: {e}. Using empty metadata.",
                file=sys.stderr,
            )
            # Backup corrupted metadata
            backup_path = METADATA_FILE.with_suffix(".json.bak")
            if METADATA_FILE.exists():
                METADATA_FILE.rename(backup_path)
            return ProfileMetadata()

    def _save_metadata(self, metadata: ProfileMetadata):
        """Save profile metadata to disk with file locking.

        Args:
            metadata: ProfileMetadata to save.
        """
        self._ensure_directories()

        # Use atomic write with temp file
        temp_file = METADATA_FILE.with_suffix(".json.tmp")

        with open(temp_file, "w") as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(
                    {"version": metadata.version, "profiles": metadata.profiles},
                    f,
                    indent=2,
                )
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        # Atomic rename
        temp_file.rename(METADATA_FILE)


def resolve_active_profile() -> str:
    """Resolve which profile to use based on priority.

    Priority (highest to lowest):
    1. HINDSIGHT_EMBED_PROFILE environment variable
    2. CLI --profile flag (from global context)
    3. Active profile from file
    4. Default (empty string)

    Returns:
        Profile name to use (empty string for default).
    """
    # 1. Environment variable
    if env_profile := os.getenv("HINDSIGHT_EMBED_PROFILE"):
        return env_profile

    # 2. CLI flag (set by caller before invoking commands)
    from . import cli

    if cli_profile := cli.get_cli_profile_override():
        return cli_profile

    # 3. Active profile file
    pm = ProfileManager()
    if active_profile := pm.get_active_profile():
        return active_profile

    # 4. Default
    return ""


def validate_profile_exists(profile: str):
    """Validate that a profile exists, exit if not.

    Args:
        profile: Profile name to validate.

    Exits:
        If profile doesn't exist, prints error and exits.
    """
    if not profile:
        # Default profile - always valid
        return

    pm = ProfileManager()
    if not pm.profile_exists(profile):
        print(
            f"Error: Profile '{profile}' not found.",
            file=sys.stderr,
        )
        print(
            f"Create it with: hindsight-embed configure --profile {profile}",
            file=sys.stderr,
        )
        sys.exit(1)
