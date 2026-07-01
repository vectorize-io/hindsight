# Troubleshooting

Common issues and solutions.

## Task Not Found

```bash
# List all available tasks
mise tasks --all

# Show task details
mise tasks <task-name>
```

## Runtime Not Found

```bash
# Install all runtimes
mise install

# Install specific runtime
mise install node@24

# List installed runtimes
mise list
```

## Port Already in Use

```bash
# Check ports
lsof -ti:8000  # API port
lsof -ti:3000  # Web port
lsof -ti:8080  # Mobile port

# Kill processes
lsof -ti:8000 | xargs kill -9
```

## Task Hangs

```bash
# Run with verbose output
mise run dev --verbose

# Debug mode
MISE_DEBUG=1 mise run dev

# Check for interactive prompts (use --yes if available)
mise run install --yes
```

## Clean State

```bash
# Stop all dev servers
pkill -f "mise run"

# Clean mise cache
mise cache clear

# Reinstall everything
mise uninstall --all
mise install
```

## Debug Configuration

```bash
# Show mise config
mise config

# Show environment
mise env

# Doctor - check for issues
mise doctor
```
