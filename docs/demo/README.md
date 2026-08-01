# AetherPdM Terminal Demos (VHS)

Scripted terminal recordings generated with [Charmbracelet VHS](https://github.com/charmbracelet/vhs).

## GIFs

| Demo | File | What it shows |
|------|------|---------------|
| Ops Loop | `ops-loop.gif` | Batch scoring → drift check → domain shift report (CWRU → Paderborn collapse) |
| Secure API | `api.gif` | Create org-scoped key → 401 without key → 200 with key |

## Prerequisites

- [Go](https://go.dev/dl/) (to install VHS)
- Linux or macOS (VHS is fragile on Windows — the CI workflow renders on `ubuntu-latest`)
- Python 3.12 + `uv` for the real CLI commands inside the tapes

## Render locally (Linux/macOS)

```bash
# Install VHS
go install github.com/charmbracelet/vhs@latest

# Make sure the vhs binary is on PATH
export PATH="$PATH:$(go env GOPATH)/bin"

# Render all tapes
vhs docs/demo/ops-loop.tape
vhs docs/demo/api.tape
```

The GIFs are written next to their tapes (e.g. `docs/demo/ops-loop.gif`).
To see real model output, first run the local setup from the root
`README.md` (download data → generate features → train models), then
start the API (`uv run uvicorn aether_pdm.serve.app:app`) before rendering
`api.tape`. The demoer is responsible for having a server up and data
available — the tapes are honest recordings, not mocks.

## Render via GitHub Actions

Push to main (or run the `demo` workflow manually): `.github/workflows/demo.yml`
renders the tapes and uploads the GIFs as artifacts.

Honest note on CI vs local rendering: the CI runner installs the repo deps
(`uv sync --extra dev --extra web`) so the commands at least attempt, but the
runner does **not** have trained models, feature parquet files, or a running
API. The CI GIFs therefore show the real workflow and command typing; the
full demo with real model output requires your local environment with trained
models. For the full experience, render locally.

## Editing

The `.tape` files are plain text. Edit the commands, then re-render.

For `api.tape`, replace the placeholder key `aether_Ab12cD34_xxxxxxxxxxxx`
with a real key from:

```bash
uv run python scripts/manage_keys.py create --name demo --org acme
# → API key (shown once): aether_Ab12cD34_xxxxxxxxxxxxxxx
```

Real keys are generated at runtime and only shown once (stored hashed).
If the placeholder is not replaced, the "200 with key" curl will fail —
that is expected and documented in the tape's header comment.

Note on the curl payload: the `[0.1]*200` in `api.tape` is **illustrative
Python list shorthand, not valid JSON**. The tape is a typed-screen recording,
so the payload is what it is on screen. If you want a real 200-sample demo,
generate a real waveform JSON array and paste it into the tape instead, e.g.:

```bash
uv run python -c "import json; print(json.dumps([0.0]*2048))"
```

The `"waveform"` field is a 1-D array of samples; anything resembling
`[0.1]*200` will be rejected by a strict JSON parser on the API side.
