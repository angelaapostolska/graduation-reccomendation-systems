# graduation-reccomendation-systems

Graduation thesis: ontology + graph-based search over a synthetic Findzzer event
dataset. Entities, taxonomy, and database are modeled on Findzzer's real production
schema; all event, venue, and performer data is synthetic.

## Pipeline

1. **`generator/`** -- generates synthetic events (with venue, performer, genre,
   party type) from Findzzer's real taxonomy, weighted by real preference data.
2. **`mapping/`** -- shapes generated records into rows matching the database schema.
3. **`db/`** -- Postgres (with `pgvector`) schema and loading scripts.
4. **`search/`** -- query script: text in, ranked events out.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your LLM API key
```

## Ontology

```
Event -[HOSTED_AT]-> Venue -[LOCATED_IN]-> City
Event -[HAS_GENRE]-> Genre
Event -[HAS_PARTY_TYPE]-> PartyType
Event -[FEATURES]-> Performer -[ASSOCIATED_WITH]-> Genre
```
