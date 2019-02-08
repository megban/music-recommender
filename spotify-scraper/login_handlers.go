package main

import (
	"log"
	"net/http"
)

func login(w http.ResponseWriter, r *http.Request) {
	redirect := auth.AuthURL(STATE)
	http.Redirect(w, r, redirect, http.StatusFound)
}

func redirectCallback(w http.ResponseWriter, r *http.Request) {
	// Check for token
	token, err := auth.Token(STATE, r)
	if err != nil {
		http.Error(w, "Failed to process Spotify token", http.StatusForbidden)
		log.Print("Could not process spotify token:", err)
	}

	// Check state is not stale
	if state := r.FormValue("state"); state != STATE {
		http.NotFound(w, r)
		log.Printf("State mismatched. Got %s, expected %s\n", state, STATE)
	}

	// Pull out a client with the token
	client := auth.NewClient(token)

	// Save the client with the userid
	user, err := client.CurrentUser()
	if err != nil {
		http.Error(w, "Failed to get user ID", http.StatusInternalServerError)
		log.Print("Could not get user id:", err)
	}
	clients[user.ID] = client

	// Put the client in a cookie
	http.SetCookie(w, &http.Cookie{
		Name:  "userid",
		Value: user.ID,
	})

	http.Redirect(w, r, "/", http.StatusFound)
}
