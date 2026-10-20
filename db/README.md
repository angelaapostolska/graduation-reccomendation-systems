# Database (Postgres + pgvector)

`schema.sql` creates the ontology tables, `load.py` loads `data/tables/*.csv`, `post_embeddings.sql` adds the vector index after embeddings exist.

## Setup

1. Install Docker Desktop (https://www.docker.com/products/docker-desktop) and pgAdmin 4 (https://www.pgadmin.org/download/).
2. Start Postgres with pgvector:

   ```
   docker run --name findzzer-db -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=findzzer_search -p 5432:5432 -d pgvector/pgvector:pg16
   ```

   Later: `docker stop findzzer-db` / `docker start findzzer-db`. The data stays in the container.
3. In pgAdmin: Register > Server. Host `localhost`, port `5432`, database `findzzer_search`, user `postgres`, password `postgres`.
4. Python 3.11+ and the dependencies: `pip install -r requirements.txt`.
5. Copy `.env.example` to `.env` and set
   `DATABASE_URL=postgresql://postgres:postgres@localhost:5432/findzzer_search`.
6. Put the table files in `data/tables/` (they are not in the repo). Build them with the pipeline in `mapping/README.md`.
7. Load: `python db/load.py --reset`. On Windows run `set PYTHONUTF8=1` first.

## Check in pgAdmin (Query Tool)

```sql
SELECT count(*) FROM events;                       -- 5876
SELECT * FROM event_details LIMIT 10;              -- events with venue, city, genres, performers
SELECT g.name, count(*) FROM event_music_genres eg JOIN music_genres g USING (music_genre_id) GROUP BY 1 ORDER BY 2 DESC;
```

## Event times

After loading (or on an existing database), run `python mapping/event_times.py --dump forge.sql.gz` and then `python db/add_times.py`. It adds `start_time`, `end_time` and `time_source` to `events` and to the `event_details` view. No embeddings change.

## Without Docker

Install PostgreSQL 16 and the pgvector extension (macOS: `brew install postgresql@16 pgvector`; Linux: `apt install postgresql-16-pgvector`). Create a database, then `CREATE EXTENSION vector;` as a superuser.

## Troubleshooting

- `extension "vector" is not available`: the server has no pgvector. Use the Docker image above.
- `port 5432 already in use`: stop the other Postgres or map another port (`-p 5433:5432`) and change the port in `DATABASE_URL`.
- `connection refused`: the container is not running (`docker start findzzer-db`).
