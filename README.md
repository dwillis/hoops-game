# Hoops 2026 (WBB Edition)

A terminal coaching simulator for women's college basketball, modeled on the
1986 Sagarin/Winston *Hoops* design and scoped to D-I WBB from 2015-16 onward.

See [HOOPS_2026_WBB.md](HOOPS_2026_WBB.md) for the design doc.

This is a non-commercial fan project. No team logos are bundled. Player names
sourced from public play-by-play feeds are used for historical-roster
realism only.

## Status

Phase 1 (ingestion + schemas). Not yet runnable end-to-end.

## Setup

```bash
uv sync --extra dev
uv run pytest
```

## Layout

- `src/hoops/` — engine, rules, simulation, UI.
- `data/rules/wbb.yaml` — per-season rules table.
- `scripts/` — ingestion and distribution-fitting scripts.
- `tests/` — unit + validation harness.
