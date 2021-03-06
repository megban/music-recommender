package main

import (
	"fmt"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/zmb3/spotify"
)

const (
	LIMIT = 50
)

func servererror(w http.ResponseWriter, msg string, err error) {
	http.Error(w, fmt.Sprintf("%s (%s)", msg, err), http.StatusInternalServerError)
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
	fmt.Fprintf(w,
		`<html><body>
		<form method=get action='/genre'>
		Genre to scrape: <input type=text name=genre>
		Max songs: <input type=number name=max>
		<input type=submit>
		</form>
		<form method=post action='/playlist' id='playlist'>
		Make playlist: <input type=text name=name>
		Song IDs:<textarea name=ids form='playlist'></textarea>
		<input type=submit>
		</form>
		<p>
		Select a playlist to download. Downloads may take several seconds.
		<ul>`)
	for _, pl := range playlists {
		fmt.Fprintf(w, "<li><a href='csv?id=%s&name=%s'>%s</a></li>", pl.ID, pl.Name, pl.Name)
	}
	fmt.Fprintf(w, "</ul></p></body></html>")
}

// Turns a playlist ID into a CSV of audio features
func (ch *ClientHandlers) DownloadCSV(w http.ResponseWriter, r *http.Request, client spotify.Client) {
	playlistID := r.FormValue("id")
	playlistName := r.FormValue("name")
	if playlistID == "" || playlistName == "" {
		servererror(w, "Missing parameters (need id, name)", nil)
		return
	}

	csv := NewTrackFeatureCSV(client, playlistName)

	for limit, offset := LIMIT, 0; ; offset += limit {
		// Get a set of tracks
		tracks, err := client.GetPlaylistTracksOpt(spotify.ID(playlistID),
			&spotify.Options{
				Limit:  &limit,
				Offset: &offset,
			}, "items")
		if err != nil {
			servererror(w, "Could not retrieve tracks in playlist", err)
			return
		}
		if len(tracks.Tracks) == 0 {
			// No tracks left
			break
		}

		// Get FullTracks to add to csv
		fulltracks := make([]spotify.FullTrack, len(tracks.Tracks))
		for i := range tracks.Tracks {
			fulltracks[i] = tracks.Tracks[i].Track
		}
		csv.Add(fulltracks)
	}

	csv.WriteHTTP(w)
}

// Scrapes a track features csv from a genre seed.
func (ch *ClientHandlers) GenreScrape(w http.ResponseWriter, r *http.Request, client spotify.Client) {
	genre := r.FormValue("genre")
	if genre == "" {
		servererror(w, "Missing genre paramater", nil)
		return
	}

	maximum, err := strconv.Atoi(r.FormValue("max"))
	if err != nil || maximum < 0 || maximum > 1e4 {
		maximum = 500
	}

	csv := NewTrackFeatureCSV(client, genre)
	for limit, offset := LIMIT, 0; limit+offset <= maximum; offset += limit {
		searchResult, err := client.SearchOpt("genre:"+genre, spotify.SearchTypeTrack, &spotify.Options{
			Limit:  &limit,
			Offset: &offset,
		})
		if err != nil {
			servererror(w, "Error searching by genre", err)
			return
		}

		if len(searchResult.Tracks.Tracks) == 0 {
			break
		}

		csv.Add(searchResult.Tracks.Tracks)
	}

	csv.WriteHTTP(w)
}

// Spectrogram creates a spectrogram of a 30 second sample for a song.
func (ch *ClientHandlers) Spectrogram(w http.ResponseWriter, r *http.Request, client spotify.Client) {
	trackID := r.FormValue("trackid")
	if trackID == "" {
		servererror(w, "Missing trackid param", nil)
		return
	}

	track, err := client.GetTrack(spotify.ID(trackID))
	if err != nil {
		servererror(w, "Could not find track info", err)
		return
	}

	preview, err := http.Get(track.PreviewURL)
	if err != nil {
		servererror(w, "Could not download preview track", err)
		return
	}

	img, err := NewSpectrogram(preview.Body)
	if err != nil {
		servererror(w, "Could not read mp3 bytes", err)
		return
	}

	png.Encode(w, img)
}

func (ch *ClientHandlers) MakePlaylist(w http.ResponseWriter, r *http.Request, client spotify.Client) {
	r.ParseForm()
	name := r.Form.Get("name")
	idstring := r.Form.Get("ids")
	if name == "" {
		servererror(w, "Missing name", nil)
		return
	}
	if idstring == "" {
		servererror(w, "Missing ids", nil)
		return
	}

	// Split by newline and remove the extra empty string
	id_split := strings.Split(idstring, "\n")
	if id_split[len(id_split)-1] == "" {
		id_split = id_split[:len(id_split)-1]
	}

	// Completely redundant, but we need to change the type.
	ids := make([]spotify.ID, len(id_split))
	for i := range id_split {
		ids[i] = spotify.ID(id_split[i])
		if len(ids[i]) > 22 {
			ids[i] = ids[i][:22]
		}
	}

	// This must succeed after going through WithClient
	cookie, _ := r.Cookie("userid")
	userid := cookie.Value

	playlist, err := client.CreatePlaylistForUser(userid, name, "Created for CS144", false)
	if err != nil {
		servererror(w, "Could not create playlist", err)
		return
	}

	for i := 0; i < len(ids); i += 100 {
		max := i + 100
		if max >= len(ids) {
			max = len(ids)
		}
		_, err = client.AddTracksToPlaylist(playlist.ID, ids[i:max]...)
		if err != nil {
			servererror(w, "Could not add tracks to playlist", err)
			return
		}
	}

	fmt.Fprintf(w, "Made playlist %s with %d songs", name, len(ids))
}

// NoClient handles loading a page that requires a client when no client is
// present. Current behaviour is to redirect to login.
func (ch *ClientHandlers) NoClient(w http.ResponseWriter, r *http.Request) {
	http.SetCookie(w, &http.Cookie{
		Name:  "callback-address",
		Value: r.URL.RequestURI(),
	})
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
	mux.HandleFunc("/genre", ch.WithClient(ch.GenreScrape))
	mux.HandleFunc("/spectrogram", ch.WithClient(ch.Spectrogram))
	mux.HandleFunc("/playlist", ch.WithClient(ch.MakePlaylist))
}

func downloadsongs(client spotify.Client, ids []spotify.ID, dir string) {
	for i := range ids {
		log.Println("Downloading", ids[i])
		track, err := client.GetTrack(ids[i])
		if err != nil {
			log.Println("Could not get track", err)
			continue
		}

		if track.PreviewURL == "" {
			log.Println("No preview for", ids[i])
			continue
		}

		preview, err := http.Get(track.PreviewURL)
		if err != nil {
			log.Println("Song download fail:", err)
			continue
		}
		defer preview.Body.Close()

		f, err := os.Create(dir + "/" + ids[i].String() + ".mp3")
		if err != nil {
			log.Println("Could not write MP3:", err)
			continue
		}
		defer f.Close()

		io.Copy(f, preview.Body)
	}
	log.Println("Done downloading MP3s")
}
