package main

import (
	"bufio"
	"flag"
	"image/png"
	"io"
	"log"
	"net/http"
	"os"
	"runtime"
	"sync"
	"time"

	"github.com/zmb3/spotify"
)

const (
	SHORTURL = "http://localhost"
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
	clients     = make(map[string]spotify.Client)
	download    = flag.String("preview-tracks", "", "download preview tracks for ids in file")
	spectrogram = flag.String("spectrogram", "", "convert ids in file from read-from to spectrograms")
	read_from   = flag.String("read-from", ".", "directory to read from")
	save_to     = flag.String("save-to", ".", "directory to output to")
)

func read_ids(filename string) []spotify.ID {
	idf, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer idf.Close()
	scanner := bufio.NewReader(idf)
	var ids []spotify.ID
	for {
		id, err := scanner.ReadString('\n')
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}
		ids = append(ids, spotify.ID(id[:len(id)-1]))
	}
	return ids
}

func main() {
	flag.Parse()

	if *download != "" {
		go func() {
			// Wait for a client from the server thread to download songs
			var client spotify.Client
			ticker := time.NewTicker(1 * time.Second)
			for len(clients) == 0 {
				<-ticker.C
			}
			for _, client = range clients {
				break
			}
			downloadsongs(client, read_ids(*download), *save_to)
		}()
		// Fall through to running server to get client
	}

	if *spectrogram != "" {
		var wg sync.WaitGroup
		in := make(chan spotify.ID)

		downloader := func() {
			wg.Add(1)
			for id := range in {
				preview, err := os.Open(*read_from + "/" + id.String() + ".mp3")
				if err != nil {
					log.Println("No mp3 file for", id)
					continue
				}
				img, err := NewSpectrogram(preview)
				if err != nil {
					log.Fatal(err)
				}

				log.Println("Write spectrogram for", id)
				w, err := os.Create(*save_to + "/" + id.String() + ".png")
				if err != nil {
					log.Fatal(err)
				}
				defer w.Close()

				png.Encode(w, img)
			}
			wg.Done()
		}

		for i := 0; i < runtime.NumCPU(); i++ {
			go downloader()
		}

		go func() {
			ids := read_ids(*spectrogram)
			for _, id := range ids {
				in <- id
			}
		}()

		wg.Wait()
		return
	}

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
