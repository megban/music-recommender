package main

import (
	_ "fmt"
	"log"
	"net/http"
	"os"

	"github.com/zmb3/spotify"
)

const (
	SHORTURL = "http://tryransom.com"
	PORT     = ""
	URL      = SHORTURL + PORT
	REDIRECT = URL + "/callback"
	STATE    = "609a8184bd694ea3dd24bff8"
)

var (
	auth = spotify.NewAuthenticator(REDIRECT,
		spotify.ScopeUserLibraryRead,
		spotify.ScopePlaylistModifyPrivate,
		spotify.ScopePlaylistReadPrivate,
		spotify.ScopePlaylistReadCollaborative,
		spotify.ScopeUserLibraryRead,
	)
	config  = struct{}{}
	clients = make(map[string]spotify.Client)
)

func main() {
	log.Print("SPOTIFY_ID=" + os.Getenv("SPOTIFY_ID"))

	mux := http.NewServeMux()
	mux.HandleFunc("/login", login)
	mux.HandleFunc("/callback", redirectCallback)

	clientHandlers := &ClientHandlers{
		Clients: clients,
	}
	clientHandlers.HandleFuncs(mux)

	server := &http.Server{
		Handler: mux,
	}
	log.Fatal(server.ListenAndServe())
}
