package main

import (
	"cifarcnn"
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	batchSize   = 1
	imgHeight   = 4
	imgWidth    = 6
	imgChannels = 1
	classes     = 3
	imgShape    = []int{batchSize, imgChannels, imgHeight, imgWidth}

	zero_image_label = tensor.New(tensor.WithShape(classes), tensor.WithBacking([]float64{1, 0, 0}))
	zero_image       = tensor.New(tensor.WithShape(imgShape...), tensor.WithBacking([]float64{
		0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
	}))

	one_image_label = tensor.New(tensor.WithShape(classes), tensor.WithBacking([]float64{0, 1, 0}))
	one_image       = tensor.New(tensor.WithShape(imgShape...), tensor.WithBacking([]float64{
		0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1,
	}))

	two_image_label = tensor.New(tensor.WithShape(classes), tensor.WithBacking([]float64{0, 0, 1}))
	two_image       = tensor.New(tensor.WithShape(imgShape...), tensor.WithBacking([]float64{
		1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1,
	}))
)

func main() {
	g := gorgonia.NewGraph()

	filters := 5
	kernelSize := 3
	depth := imgChannels
	cnn := cifarcnn.BuildConvnet(g, filters, kernelSize, depth)
	fmt.Println(cnn)

}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }
