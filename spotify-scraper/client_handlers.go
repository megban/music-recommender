package main

import (
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/zmb3/spotify"
)

const (
	LIMIT = 50
)

func servererror(w http.ResponseWriter, msg string, err error) {
	w.WriteHeader(http.StatusInternalServerError)
	fmt.Fprintf(w, "500: Internal error -- %s (%s)", msg, err)
}

// whois returns the name of the user for a client
func whois(client spotify.Client) string {
	user, err := client.CurrentUser()
	if err != nil {
		return "Anonymous User"
	} else {
		return user.DisplayName
	}
}

// ClientHandler is an http.Handler with a Spotify client attached.
type ClientHandler func(w http.ResponseWriter, r *http.Request, client spotify.Client)

// ClientHandlers has all handlers that use clients as methods on it.
type ClientHandlers struct {
	// Clients keeps a map of user IDs to active clients registered with Spotify
	Clients map[string]spotify.Client
}

// Displays a list of playlists to download.
func (ch *ClientHandlers) Landing(w http.ResponseWriter, r *http.Request, client spotify.Client) {
	var playlists []spotify.SimplePlaylist

	// Pull down a list of playlists 50 at a time
	for limit, offset := LIMIT, 0; ; offset += limit {
		shortlist, err := client.CurrentUsersPlaylistsOpt(&spotify.Options{
			Limit:  &limit,
			Offset: &offset,
		})
		if err != nil {
			servererror(w, "Could not find your playlists", err)
			return
		}
		if len(shortlist.Playlists) == 0 {
			// No more tracks
			break
		}

		playlists = append(playlists, shortlist.Playlists...)
	}

	// Generate html. TODO Use a template system -- vulnerable code
	fmt.Fprintf(w, "<html><body>Select a playlist to download. Downloads may take several seconds.<ul>")
	for _, pl := range playlists {
		fmt.Fprintf(w, "<li><a href='csv?id=%s&name=%s'>%s</a></li>", pl.ID, pl.Name, pl.Name)
	}
	fmt.Fprintf(w, "</ul></body></html>")
}

// Turns a playlist ID into a CSV of audio features
func (ch *ClientHandlers) DownloadCSV(w http.ResponseWriter, r *http.Request, client spotify.Client) {
	playlistID := r.FormValue("id")
	playlistName := r.FormValue("name")
	if playlistID == "" || playlistName == "" {
		fmt.Fprintf(w, "Missing paramaters (need id, name)")
		return
	}

	// Assuming status OK for efficiency
	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition", "attachment; filename=\""+playlistName+".csv\"")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "id,name,album,acousticness,danceability,energy,instrumentalness,liveness,loudness,speechiness,valence,duration,key,mode,tempo,time_sig\n")

	for limit, offset := LIMIT, 0; ; offset += limit {
		// Get a set of tracks
		tracks, err := client.GetPlaylistTracksOpt(spotify.ID(playlistID),
			&spotify.Options{
				Limit:  &limit,
				Offset: &offset,
			}, "items")
		if err != nil {
			fmt.Fprintf(w, "Error occurred, data incomplete")
			return
		}
		if len(tracks.Tracks) == 0 {
			// No tracks left
			break
		}

		var idList []spotify.ID
		for _, track := range tracks.Tracks {
			idList = append(idList, track.Track.ID)
		}

		// Get features of those tracks
		features, err := client.GetAudioFeatures(idList...)
		if err != nil {
			fmt.Fprintf(w, "Error occurred, data incomplete")
			return
		}

		// Write features
		for i, f := range features {
			if f == nil {
				// Skip missing songs
				continue
			}

			// Print the CSV item. Strings w/ quotations have to be fixed. See
			// above line for format.
			fmt.Fprintf(w, "%s,\"%s\",\"%s\",%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d,%f,%d\n",
				f.ID.String(),
				strings.Replace(tracks.Tracks[i].Track.Name, "\"", "\"\"", -1),
				strings.Replace(tracks.Tracks[i].Track.Album.Name, "\"", "\"\"", -1),
				f.Acousticness,
				f.Danceability,
				f.Energy,
				f.Instrumentalness,
				f.Liveness,
				f.Loudness,
				f.Speechiness,
				f.Valence,
				f.Duration,
				f.Key,
				f.Mode,
				f.Tempo,
				f.TimeSignature,
			)
		}
	}
}

// NoClient handles loading a page that requires a client when no client is
// present. Current behaviour is to redirect to login.
func (ch *ClientHandlers) NoClient(w http.ResponseWriter, r *http.Request) {
	http.Redirect(w, r, "/login", http.StatusFound)
}

// WithClient decorates a ClientHandler to automatically detect the client.
func (ch *ClientHandlers) WithClient(next ClientHandler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		cookie, err := r.Cookie("userid")
		if err != nil {
			ch.NoClient(w, r)
			return
		}

		client, ok := ch.Clients[cookie.Value]
		if !ok {
			http.SetCookie(w, &http.Cookie{
				Name:   "userid",
				Value:  "INVALID",
				MaxAge: -1,
			})
			ch.NoClient(w, r)
			return
		}

		log.Println(r.Method, r.URL, "accessed by", whois(client))
		next(w, r, client)
	}
}

// HandleFuncs registers the client endpoints to a mux.
func (ch *ClientHandlers) HandleFuncs(mux *http.ServeMux) {
	mux.HandleFunc("/", ch.WithClient(ch.Landing))
	mux.HandleFunc("/csv", ch.WithClient(ch.DownloadCSV))
}
