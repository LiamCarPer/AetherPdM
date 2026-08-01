# AetherPdM — Secure API Demo (VHS tape)
# Renders: docs/demo/api.gif
# Commands shown are the real AetherPdM CLI + curl flow:
#   manage_keys.py create   -> create an org-scoped API key (shown once)
#   export AETHER_API_KEY_AUTH_ENABLED=true -> enable auth (documented env var)
#   curl (no key)           -> 401 Unauthorized
#   curl (with key)         -> 200 OK
#
# IMPORTANT: the key in the last curl is a PLACEHOLDER
# (aether_Ab12cD34_xxxxxxxxxxxx). Real keys are generated at runtime and shown
# once. Edit this tape and paste a real key from:
#   uv run python scripts/manage_keys.py create --name demo --org acme
# See docs/demo/README.md -> "Editing".

Output docs/demo/api.gif
Set FontSize 13
Set Width 1200
Set Height 700
Set Padding 24
Set TypingSpeed 40ms

Type "echo 'AetherPdM — Secure API Demo'"
Enter
Sleep 500ms
Ctrl+L

# 1. Create an org-scoped key
Type "uv run python scripts/manage_keys.py create --name plant-1 --org acme"
Enter
Sleep 2s
Ctrl+L

# 2. Enable auth (documented env var)
Type "export AETHER_API_KEY_AUTH_ENABLED=true"
Enter
Sleep 500ms

# 3. Score without a key → 401
Type `curl -s -X POST http://localhost:8000/v1/assets/motor-001/score -H 'Content-Type: application/json' -d '{"waveform": [0.1]*200, "sampling_rate": 12000}'`
Enter
Sleep 2s
Ctrl+L

# 4. Score WITH the key → 200
Type `curl -s -X POST http://localhost:8000/v1/assets/motor-001/score -H 'Content-Type: application/json' -H 'X-API-Key: aether_Ab12cD34_xxxxxxxxxxxx' -d '{"waveform": [0.1]*200, "sampling_rate": 12000}'`
Enter
Sleep 3s
