CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE cities (
    city_id  integer PRIMARY KEY,
    name     text NOT NULL,
    country  text NOT NULL,
    UNIQUE (name, country)
);

CREATE TABLE business_types (
    type_id              integer PRIMARY KEY,
    name                 text NOT NULL UNIQUE,
    is_project_addition  boolean NOT NULL
);

CREATE TABLE venues (
    venue_id       text PRIMARY KEY,
    name           text NOT NULL,
    city_id        integer NOT NULL REFERENCES cities,
    instagram_url  text,
    is_fictional   boolean NOT NULL
);

CREATE TABLE venue_types (
    venue_id  text NOT NULL REFERENCES venues ON DELETE CASCADE,
    type_id   integer NOT NULL REFERENCES business_types,
    PRIMARY KEY (venue_id, type_id)
);

CREATE TABLE music_genres (
    music_genre_id  integer PRIMARY KEY,
    name      text NOT NULL UNIQUE
);

CREATE TABLE event_types (
    event_type_id  integer PRIMARY KEY,
    name           text NOT NULL UNIQUE
);

CREATE TABLE performers (
    performer_id  text PRIMARY KEY,
    name          text NOT NULL,
    kind          text,
    source        text NOT NULL
);

CREATE TABLE performer_music_genres (
    performer_id  text NOT NULL REFERENCES performers ON DELETE CASCADE,
    music_genre_id      integer NOT NULL REFERENCES music_genres,
    event_count   integer NOT NULL,
    PRIMARY KEY (performer_id, music_genre_id)
);

CREATE TABLE events (
    event_id            text PRIMARY KEY,
    source_event_id     text NOT NULL UNIQUE,
    origin              text NOT NULL,
    is_synthetic        boolean NOT NULL,
    name                text NOT NULL,
    description         text NOT NULL,
    description_source  text NOT NULL,
    event_date          date NOT NULL,
    venue_id            text NOT NULL REFERENCES venues,
    entrance_mkd        integer,
    entrance_source     text,
    theme               text,
    dress_code          text,
    promotion           text,
    age_limit           text,
    language            text,
    includes            text,
    music_genre_source        text NOT NULL,
    synthetic_fields    text,
    embedding           vector(1536)
);

CREATE TABLE event_music_genres (
    event_id  text NOT NULL REFERENCES events ON DELETE CASCADE,
    music_genre_id  integer NOT NULL REFERENCES music_genres,
    source    text NOT NULL,
    PRIMARY KEY (event_id, music_genre_id)
);

CREATE TABLE event_event_types (
    event_id       text NOT NULL REFERENCES events ON DELETE CASCADE,
    event_type_id  integer NOT NULL REFERENCES event_types,
    PRIMARY KEY (event_id, event_type_id)
);

CREATE TABLE event_performers (
    event_id      text NOT NULL REFERENCES events ON DELETE CASCADE,
    performer_id  text NOT NULL REFERENCES performers,
    PRIMARY KEY (event_id, performer_id)
);

CREATE INDEX events_venue_idx ON events (venue_id);
CREATE INDEX events_date_idx ON events (event_date);
CREATE INDEX venues_city_idx ON venues (city_id);
CREATE INDEX event_music_genres_genre_idx ON event_music_genres (music_genre_id);
CREATE INDEX event_event_types_type_idx ON event_event_types (event_type_id);
CREATE INDEX event_performers_performer_idx ON event_performers (performer_id);
CREATE INDEX events_text_idx ON events USING gin (to_tsvector('english', name || ' ' || description));

CREATE VIEW event_details AS
SELECT e.event_id,
       e.name,
       e.description,
       e.event_date,
       v.name AS venue,
       c.name AS city,
       (SELECT string_agg(bt.name, ', ' ORDER BY bt.name)
          FROM venue_types vt JOIN business_types bt USING (type_id) WHERE vt.venue_id = v.venue_id) AS venue_types,
       (SELECT string_agg(g.name, ', ' ORDER BY g.name)
          FROM event_music_genres eg JOIN music_genres g USING (music_genre_id) WHERE eg.event_id = e.event_id) AS music_genres,
       (SELECT string_agg(pt.name, ', ' ORDER BY pt.name)
          FROM event_event_types ep JOIN event_types pt USING (event_type_id) WHERE ep.event_id = e.event_id) AS event_types,
       (SELECT string_agg(p.name, ', ' ORDER BY p.name)
          FROM event_performers pf JOIN performers p USING (performer_id) WHERE pf.event_id = e.event_id) AS performers,
       e.entrance_mkd,
       e.theme,
       e.promotion,
       e.is_synthetic
FROM events e
JOIN venues v USING (venue_id)
JOIN cities c USING (city_id);
