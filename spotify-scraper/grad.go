package main

import (
	"fmt"
	"image/color"
	"math"
)

type Gradient []color.RGBA

// At returns the color in a gradient for a specific intensity.
func (g Gradient) At(intensity float64) color.RGBA {
	if intensity < 0 {
		return g[0]
	}
	if intensity > 1 {
		return g[len(g)-1]
	}

	n := float64(len(g)) - 1

	// Indices for the colors we are lerping between
	li := math.Floor(intensity * n)
	ri := math.Ceil(intensity * n)

	// Factor of each color
	lf := intensity*n - li
	rf := ri - intensity*n

	defer func() {
		e := recover()
		if e != nil {
			fmt.Println("intensity is", intensity)
			fmt.Println("li and ri are", li, ri)
			panic(e)
		}
	}()

	l, r := g[int(li)], g[int(ri)]

	return color.RGBA{
		R: uint8(float64(l.R)*lf) + uint8(float64(r.R)*rf),
		G: uint8(float64(l.G)*lf) + uint8(float64(r.G)*rf),
		B: uint8(float64(l.B)*lf) + uint8(float64(r.B)*rf),
		A: uint8(float64(l.A)*lf) + uint8(float64(r.A)*rf),
	}
}
