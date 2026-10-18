# Mapping pipeline

Run from the repository root. All inputs and outputs live in the gitignored `data/` and `generator/data/` folders.

1. `python mapping/real_events.py base --dump forge.sql.gz` then `performers`, then `descriptions`: real events at the curated venues, rewritten.
2. `python mapping/synthetic_events.py`: synthetic workshops, meetups, conferences and comedy.
3. `python mapping/synthetic_gap_events.py`: synthetic concerts and rare genres.
4. `python mapping/merge_events.py`: writes `data/events_final.csv`.
5. `python mapping/build_tables.py`: writes the normalized tables to `data/tables/`.

Then load them with `python db/load.py --reset` (see `db/README.md`).
