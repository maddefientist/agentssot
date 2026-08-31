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
   Edit `.env`, set `POSTGRES_PASSWORD`, and generate a one-time bootstrap key:
   ```bash
   python -c 'import secrets; print("ssot_" + secrets.token_urlsafe(32))'
   ```
   Store that value as `BOOTSTRAP_ADMIN_API_KEY`. The service hashes it and
   never writes the plaintext to logs.

3. **Start the stack:**
   ```bash
   docker compose up -d --build
   ```

4. **After first successful startup**, remove `BOOTSTRAP_ADMIN_API_KEY` from
   `.env` and retain its plaintext only in your secret manager. Existing
   databases that already contain an API key do not need it again.

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

Run the non-live test suite before opening a pull request:

```bash
cd api
python -m pip install -r requirements.txt
DATABASE_URL=postgresql+psycopg://test:test@127.0.0.1/test \
  python -m pytest -q -m "not integration and not smoke"
```

Tests marked `integration` or `smoke` require an explicitly configured service;
the default test command never contacts a production Hive.

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Make your changes with clear, descriptive commits.
3. Run the non-live test suite and ensure the stack starts cleanly.
4. Test the behavior that would fail if your change were wrong; `/health` alone
   is not functional verification.
5. Open a PR with a description of what changed and why.

## Code Style

- Python: follow existing patterns in `api/app/` and keep tests close to the behavior they pin.
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
