# Open-source Release Manifest

This manifest defines the public sharing surface. It intentionally excludes local
runtime data, private reports, internal design history, credentials and machine-
specific deployment records.

## Included

```text
README.md
LICENSE
pyproject.toml
.env.example
.gitignore
src/academic_landscape/
tests/
docs/GETTING_STARTED.md
docs/OPEN_SOURCE_PROJECT_GUIDE_FOR_HUMANS.md
docs/OPEN_SOURCE_PROJECT_SPEC_FOR_AI.md
docs/OPEN_SOURCE_RELEASE_MANIFEST.md
docs/HUMAN_READABLE_REPORT_TEMPLATE.md
docs/PIPELINE_ARCHITECTURE.html
docs/ARCHITECTURE_20260927.html
docs/ARCHITECTURE_FLOW_20260927.html
```

## Excluded

```text
.env.local and all non-example .env files
runtime/ and generated reports/databases/vector stores
.venv/ build/ dist/ *.egg-info/
private/internal planning and audit documents
local backups, caches, logs and provider credentials
```

## Public-release guarantees

Before sharing a copy, verify that the export contains:

- no real API key, password, token, cookie or private certificate;
- no absolute path to a developer machine;
- no private names, chat identifiers or internal service addresses;
- no corpus PDFs, extracted pages, evidence databases or generated reports;
- no ignored local environment file;
- no unreviewed runtime artifact.

All service endpoints and credentials must be supplied by the person deploying the
project. The public example configuration contains placeholders only.

## Rebuild the public export

From the repository root, copy only the allowlisted paths above into a new
release directory, then run the scans below. Do not use `git add -A` as a release
method: the working tree may contain local runtime artifacts and ignored secrets.

```bash
python -m pip wheel --no-deps -w dist-public/ .
academic-landscape doctor
python -m pytest tests/
ruff check src tests
```

## Secret and machine-path scan

```bash
grep -RInE 'sk-[A-Za-z0-9]{20,}|api[_-]?key[[:space:]]*[:=][[:space:]]*[^<`[:space:]]{12,}|password[[:space:]]*[:=][[:space:]]*[^<`[:space:]]{8,}' \
  README.md .env.example src tests docs/OPEN_SOURCE* docs/GETTING_STARTED.md \
  || true
```

A scan hit must be classified before release. Environment-variable names,
placeholder values and generic `127.0.0.1` test fixtures are not credentials, but
real values and private machine paths are release blockers. Run an additional
absolute-path scan over the allowlisted export using the release tooling of your
choice; keep that scan outside the exported source if its patterns embed local
machine names.

## License

MIT. The recipient must also review the licenses of optional dependencies before
redistributing a built environment.
