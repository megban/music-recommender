# Spotify Scraper

This is a simple web server to download the audio features of songs (formatted
as a CSV) for a given playlist.

## Build & Run

 - Create a Spotify developer account and make an "app"
 - Edit some of the constants in `main.go` (SHORTURL and STATE) (TODO these
	 should really be env vars or detected/generated at runtime)
 - Building with Go 1.11+ (with module support) should work as expected
 - Set your Spotify app's ID as SPOTIFY_ID in your environment
 - Optional: Set cap_net_bind_service=ep on the binary to run as normal user
	 (root needed for port 80 otherwise -- Spotify seems not to like callback
	 links that aren't on port 80/443)
 - `./spotify-scraper`
