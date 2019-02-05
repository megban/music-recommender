package main

import (
	"bytes"
	"fmt"
	"net/http"
	"strings"

	"github.com/zmb3/spotify"
)

type TrackFeatureCSV struct {
	client spotify.Client
	buffer *bytes.Buffer
	name   string
}

func NewTrackFeatureCSV(client spotify.Client, name string) *TrackFeatureCSV {
	return &TrackFeatureCSV{client, &bytes.Buffer{}, name}
}

func (csv *TrackFeatureCSV) WriteHTTP(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition", "attachment; filename=\""+csv.name+".csv\"")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("id,name,album,acousticness,danceability,energy,instrumentalness,liveness,loudness,speechiness,valence,duration,key,mode,tempo,time_sig\n"))
	w.Write(csv.buffer.Bytes())
}

func (csv *TrackFeatureCSV) Add(tracks []spotify.FullTrack) error {
	// Retrieve a list of IDs to get features
	idList := make([]spotify.ID, len(tracks))
	for i := range tracks {
		idList[i] = tracks[i].ID
	}

	// Get features of those tracks
	features, err := csv.client.GetAudioFeatures(idList...)
	if err != nil {
		return err
	}

	// Write features
	for i, f := range features {
		if f == nil {
			// Skip missing songs
			continue
		}

		// Print the CSV item. Strings w/ quotations have to be fixed.
		// See header for format.
		fmt.Fprintf(csv.buffer,
			"%s,\"%s\",\"%s\",%f,%f,%f,%f,%f,%f,%f,%f,%d,%d,%d,%f,%d\n",
			f.ID.String(),
			strings.Replace(tracks[i].Name, "\"", "\"\"", -1),
			strings.Replace(tracks[i].Album.Name, "\"", "\"\"", -1),
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

	return nil
}
