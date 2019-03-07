package main

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"math"
	"math/cmplx"

	"github.com/faiface/beep/mp3"
	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/window"
)

const (
	WINDOW = 1024
	SLIDE  = 1024
)

var (
	WINDOWCONV = window.Hamming(WINDOW)
	GRAD       = Gradient{
		color.RGBA{0x00, 0x00, 0x00, 0xff},
		color.RGBA{0x00, 0x00, 0xFF, 0xff},
		color.RGBA{0x00, 0xFF, 0xFF, 0xff},
		color.RGBA{0x00, 0xFF, 0x00, 0xff},
		color.RGBA{0xff, 0xff, 0x00, 0xff},
		color.RGBA{0xff, 0x00, 0x00, 0xff},
	}
)

func prod(x, y []float64) []float64 {
	if len(x) != len(y) {
		return nil
	}
	z := make([]float64, len(x))
	for i := range z {
		z[i] = x[i] * y[i]
	}
	return z
}

// Spectrogram is constructed using an mp3 source file and satisfies the
// image.Image interface, outputting a spectrogram of the image.
// MP3 format is 16 bit little endian, 2 channels. That's 4 bytes for every
// sample.
type Spectrogram struct {
	mag [][]float64
	max float64
}

func NewSpectrogram(src io.ReadCloser) (*Spectrogram, error) {
	// Create the decoder to get audio bytes
	dec, _, err := mp3.Decode(src)
	if err != nil {
		return nil, err
	}
	defer dec.Close()

	// Read out audio
	wavebuf := make([][2]float64, dec.Len())
	n, ok := dec.Stream(wavebuf)
	if !ok || n != len(wavebuf) {
		return nil, errors.New("Could not decode mp3")
	}

	// Filter the two channels into one
	maxchan := make([]float64, len(wavebuf))
	for i := range maxchan {
		maxchan[i] = wavebuf[i][0]
		if wavebuf[i][1] > maxchan[i] {
			maxchan[i] = wavebuf[i][1]
		}
	}

	width := len(maxchan) / SLIDE
	if width > 8192 {
		width = 8192
	}
	buf := make([][]complex128, width)
	defer func() {
		if e := recover(); e != nil {
			fmt.Println("width is", width)
			panic(e)
		}
	}()
	for i := range buf {
		buf[i] = fft.FFTReal(prod(maxchan[i*SLIDE:i*SLIDE+WINDOW], WINDOWCONV))
	}

	var magnitudes [][]float64
	max := math.Inf(-1)
	for i := range buf {
		var samplemag []float64
		for j := range buf[i] {
			mag, _ := cmplx.Polar(buf[i][j])
			mag = math.Log(mag)
			samplemag = append(samplemag, mag)
			if mag > max {
				max = mag
			}
		}
		magnitudes = append(magnitudes, samplemag)
	}

	return &Spectrogram{magnitudes, max}, nil
}

func (s *Spectrogram) ColorModel() color.Model {
	return color.RGBAModel
}

func (s *Spectrogram) Bounds() image.Rectangle {
	return image.Rectangle{
		Min: image.Point{0, 0},
		Max: image.Point{2 * len(s.mag), WINDOW / 2},
	}
}

func (s *Spectrogram) At(x, y int) color.Color {
	return s.UpsidedownLog(x/2, (WINDOW/2)-y)
}

func (s *Spectrogram) UpsidedownLog(x, y int) color.Color {
	return GRAD.At(s.norm(x, int(logscale(float64(y), WINDOW/2))))
}

func (s *Spectrogram) Upsidedown(x, y int) color.Color {
	return GRAD.At(s.norm(x, y))
}

func logscale(i, tot float64) float64 {
	logmax := math.Log(tot+1) / math.Log(2)
	exp := logmax * i / tot
	return math.Round(math.Pow(2, exp) - 1)
}

func (s *Spectrogram) norm(x, y int) float64 {
	return s.mag[x][y] / s.max
}
