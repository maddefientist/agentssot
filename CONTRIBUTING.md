# Contributing to AgentSSOT

Thanks for your interest in contributing! This project is open to contributions of all kinds.

## Local Development Setup

1. **Clone the repo:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/agentssot.git
   cd agentssot
   ```

2. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set at least `POSTGRES_PASSWORD`.

3. **Start the stack:**
   ```bash
   docker compose up -d --build
   ```

4. **Capture the bootstrap admin key** (printed once on first startup):
   ```bash
   docker compose logs api | grep BOOTSTRAP_ADMIN_API_KEY
   ```

5. **Open the dashboard:**
   - `http://localhost:8088/`
   - Swagger docs: `http://localhost:8088/docs`

## Making Changes

- The API lives in `api/app/`. It's a standard FastAPI application.
- Database migrations are in `db/init/`. The schema is bootstrapped on first run.
- The web UI is plain HTML/CSS/JS in `api/app/ui/` (no build step).

After making code changes to the API:
```bash
docker compose up -d --build api
```

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Make your changes with clear, descriptive commits.
3. Ensure the stack starts cleanly (`docker compose up -d --build`).
4. Test that `/health` returns `{"status": "ok", ...}`.
5. Open a PR with a description of what changed and why.

## Code Style

- Python: follow existing patterns in `api/app/`. No strict formatter enforced yet, but keep it consistent.
- SQL: keep `db/init/` scripts idempotent (`CREATE TABLE IF NOT EXISTS`, etc.).
- JS/HTML/CSS: keep the UI lightweight. No build tools, no frameworks.

## Reporting Issues

Open a GitHub issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Your environment (OS, Docker version, etc.)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
