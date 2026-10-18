# generator/data (local only)

Everything in this folder except this README is gitignored on purpose: the venue
and performer lists are real public business and artist information and are not
published here.

Files the generator expects, in this folder:

- `venues.csv`: `venue_id,name,instagram_url,location_types,city,country,is_fictional`
  (`location_types` is pipe-separated; may include Beach Bar, Theater, Workshop Space, Coworking Space, Event Space)
- `performers.csv`: `performer_id,name,kind,genres,source,genres_confirmed`
  (`genres` is pipe-separated)

Without these files the generator cannot run.
