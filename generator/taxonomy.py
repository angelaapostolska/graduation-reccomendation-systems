"""
Findzzer's real taxonomy, pulled from the actual production lists (Tier 1 backend
data). This is the authoritative vocabulary for every generated event -- no
invented categories, same values the real system uses.
"""

MUSIC_GENRES = [
    "Retro", "Rock", "Jazz", "Pop", "Hip-Hop", "R&B", "Blues", "Folk", "Metal",
    "Punk", "Indie", "Soul", "Funk", "Gospel", "Rap", "Ska", "Alternative",
    "Trap", "Electronic", "Disco", "Techno", "House", "Trance", "Balkan Party",
    "Ex-Yu Hits", "Macedonian", "Reggae", "Classical", "Opera", "Dancehall",
    "Latin", "Bluegrass", "Salsa", "Vinyl", "Gig", "Country", "Greek Music",
    "Afro House",
]

PARTY_TYPES = [
    "Private Party", "Festival", "Disco", "Boat Party", "Caffe Bar",
    "Karaoke Night", "Beach Party", "Concert", "Bar", "Silent Disco",
    "Pool Party", "Graduation Party", "BBQ Party", "Bonfire Party",
    "Masquerade", "Restaurant", "Standup Comedy", "Open Day", "Sport Event",
    "E-Sports", "Park Event", "Outdoor Cinema", "Race", "Theater",
    "New Year Party", "Convention", "Wine Fest", "Beer Fest", "Other",
    "Car Meet", "Protest", "Birthday Party", "House Party", "Themed Party",
    "Dj Set", "Farmers Market", "Handmade Market", "Poetry", "Dance Show",
    "Outdoor Adventure", "Social Gathering", "University Event", "Charity",
    "Networking", "Workshop", "Opera & Ballet", "Marathon", "Martial Arts",
    "Yoga Class", "Live Music",
]

LOCATION_TYPES = [
    "Concert Venue", "Restaurant", "Cafe", "Fast Food", "Fine Dining",
    "Pizzeria", "Bakery", "Bar", "Pub", "Nightclub", "Diner", "Food Truck",
    "Buffet", "Steakhouse", "Sushi Bar", "Coffee Shop", "Tea House",
    "Ice Cream Parlor", "Bistro", "Lounge", "Cocktail Bar", "Sports Bar",
    "Karaoke Bar", "Wine Bar", "Dive Bar", "Gastropub", "Cafeteria", "Disco",
    "Live Music Venue", "Irish Pub", "Tapas Bar",
]

# Project additions: venue types that exist in the real Findzzer venue list but
# are missing from Findzzer's 31-type taxonomy (confirmed by the user).
EXTRA_LOCATION_TYPES = ["Beach Bar", "Theater", "Workshop Space", "Coworking Space", "Event Space"]
ALL_LOCATION_TYPES = LOCATION_TYPES + EXTRA_LOCATION_TYPES

# Party types that plausibly carry a performer/artist (DJ, band, comedian, host).
# Non-performer party types (e.g. Farmers Market, Marathon) never get one.
PERFORMER_ELIGIBLE_PARTY_TYPES = {
    "Disco", "Concert", "Dj Set", "Live Music", "Karaoke Night",
    "Standup Comedy", "Silent Disco", "Festival", "Dance Show",
    "Opera & Ballet", "Theater", "New Year Party",
}
