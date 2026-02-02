# Contributing to Hindsight

Thanks for your interest in contributing to Hindsight!

## Getting Started

1. Fork and clone the repository
   ```bash
   git clone git@github.com:vectorize-io/hindsight.git
   cd hindsight
   ```
2. Set up your environment:
   ```bash
   cp .env.example .env
   ```
   Edit the .env to add LLM API key and config as required

3. Install dependencies:
   ```bash
   # Python dependencies
   uv sync --directory hindsight-api/

   # Node dependencies (uses npm workspaces)
   npm install
   ```

## Development

### Running the API locally

```bash
./scripts/dev/start-api.sh
```

### Running the Control Plane locally

```bash
./scripts/dev/start-control-plane.sh
```

### Running the documentation locally

```bash
./scripts/dev/start-docs.sh
```

### Running tests

```bash
cd hindsight-api
uv run pytest tests/
```

### Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for Python linting and formatting, and ESLint/Prettier for TypeScript.

#### Setting up git hooks (recommended)

Set up git hooks to automatically lint and format code before each commit:

```bash
./scripts/setup-hooks.sh
```

This configures git to use the hooks in `.githooks/`, which run all scripts in `scripts/hooks/` on commit. The lint hook runs in parallel:
- **Python**: `ruff check --fix`, `ruff format`, `ty check`
- **TypeScript**: `eslint --fix`, `prettier`

#### Manual linting and formatting

```bash
# Run all lints (same as pre-commit)
./scripts/hooks/lint.sh

# Or run individually for Python:
cd hindsight-api
uv run ruff check --fix .   # Lint and auto-fix
uv run ruff format .        # Format code
uv run ty check hindsight_api  # Type check
```

#### Style guidelines

- Use Python type hints
- Follow existing code patterns
- Keep functions focused and well-named

## Pull Requests

1. Create a feature branch from `main`
2. Make your changes
3. Run tests to ensure nothing breaks
4. Submit a PR with a clear description of changes

## Release Process

The project uses `scripts/release.sh` for creating releases. This script automates the entire release workflow:

1. Bumps version in all components (API, clients, CLI, control plane, Helm)
2. **Regenerates OpenAPI spec and client SDKs** (Python, TypeScript, Rust)
3. Updates documentation versioning
4. Creates a commit and git tag
5. Pushes to GitHub (triggers CI/CD to publish packages)

### Usage

```bash
./scripts/release.sh <version>
```

**Example:**
```bash
./scripts/release.sh 0.5.0
```

### Important for Developers

- During development, version bumps in `__init__.py` do NOT require client regeneration
- Clients are only regenerated during releases
- Do not manually run `./scripts/generate-clients.sh` unless testing generation changes
- Client version comments will reflect the API version from the latest release

## Reporting Issues

Open an issue on GitHub with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version)

## Questions?

Open a discussion on GitHub or reach out to the maintainers.
