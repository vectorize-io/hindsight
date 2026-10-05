# Contributing to Hindsight

Thanks for your interest in Hindsight!

## How to contribute

**We don't accept pull requests from outside the Hindsight team.** PRs opened by
external contributors are closed without review. This is not about you or the
quality of your code: it lets the team decide what gets built, review fewer
half-finished changes, and ship fixes together.

The best way to help is to tell us what's wrong or missing:

- **Bugs**: [open a bug report](https://github.com/vectorize-io/hindsight/issues/new?template=bug_report.yml).
  A clear way to reproduce it is the most useful thing you can give us. If you
  already know where the problem is in the code, say so in the issue.
- **Features, new integrations, extensions and bank templates**:
  [open a feature request](https://github.com/vectorize-io/hindsight/issues/new?template=feature_request.yml).
- **Questions and ideas**: use [GitHub Discussions](https://github.com/vectorize-io/hindsight/discussions).
- **Security issues**: follow [SECURITY.md](./SECURITY.md). Don't open a public issue.

The rest of this guide covers building and running Hindsight from source, which
is useful for reproducing a bug before you report it.

## Getting Started

1. Clone the repository
   ```bash
   git clone git@github.com:vectorize-io/hindsight.git
   cd hindsight
   ```

2. Bootstrap your dev environment in one shot:
   ```bash
   ./scripts/dev/setup.sh
   ```
   This is idempotent (safe to re-run) and gets you ready to develop, including
   offline. It:
   - installs the required toolchains if missing (uv/Python, Node/npm, Rust/cargo),
   - creates `.env` from `.env.example` (remember to add your LLM API key),
   - configures git hooks,
   - installs all Python and Node workspace dependencies,
   - pre-downloads the local ML models + tokenizer so the API runs offline,
   - builds the TypeScript SDK and the Rust CLI.

   Useful flags: `--skip-build` (deps only), `--skip-models` (skip ML model
   download), `--with-docs` (also build the docs site), `--force` (rebuild
   artifacts). Docker image builds are out of scope. Run
   `./scripts/dev/setup.sh --help` for details.

### Manual setup

If you'd rather set things up by hand instead of running the script above:

1. Set up your environment:
   ```bash
   cp .env.example .env
   ```
   Edit the .env to add LLM API key and config as required

2. Install dependencies:
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
- Environment details (OS, Python version, Hindsight version, LLM provider)

## Questions?

Open a discussion on GitHub or reach out to the maintainers.
