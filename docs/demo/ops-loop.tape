# AetherPdM — Autonomous Ops Loop Demo (VHS tape)
# Renders: docs/demo/ops-loop.gif
# Commands shown are the real AetherPdM CLIs:
#   run_ops_pipeline.py  -> batch score + drift check (retrain skipped for demo)
#   run_domain_shift.py  -> CWRU (source) vs Paderborn (target) drift study
#   cat ... | head -40   -> preview the generated Markdown report

Output docs/demo/ops-loop.gif
Set FontSize 13
Set Width 1200
Set Height 700
Set Padding 24
Set TypingSpeed 40ms

Type "echo 'AetherPdM — Autonomous Ops Loop Demo'"
Enter
Sleep 500ms
Type "echo '====================================='"
Enter
Sleep 500ms
Ctrl+L

# 1. Show the batch scorer (real CLI)
Type "uv run python scripts/run_ops_pipeline.py --features data/interim/features/features_v1.parquet --org acme --no-retrain"
Enter
Sleep 3s
Ctrl+L

# 2. Show the domain shift study (the wow moment)
# Note: --cwru-features uses the ACTUAL repo path where the CWRU features
# live (data/interim/features/features_v1.parquet) — the same path the
# ops pipeline references. A local render needs both parquet files to exist.
Type "uv run python scripts/run_domain_shift.py --cwru-features data/interim/features/features_v1.parquet --paderborn-features data/interim/paderborn/features_v1.parquet --output reports/domain-shift-cwru-to-paderborn.md"
Enter
Sleep 4s
Ctrl+L

# 3. Show the drift report
Type "cat reports/domain-shift-cwru-to-paderborn.md | head -40"
Enter
Sleep 3s
