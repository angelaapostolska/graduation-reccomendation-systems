CREATE INDEX IF NOT EXISTS events_embedding_idx ON events USING hnsw (embedding vector_cosine_ops);
