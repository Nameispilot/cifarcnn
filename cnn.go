package cifarcnn

import (
	"fmt"

	"github.com/pkg/errors"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type convnet struct {
	w0, w1, w2, w3, w4 *gorgonia.Node // layers weights
	d0, d1, d2, d3     float64        // dropout probabilities

	out *gorgonia.Node
}

func NewCNN(g *gorgonia.ExprGraph, batchSize, kernelSize, depth, outputSize int) *convnet {
	w0 := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(batchSize, depth, kernelSize, kernelSize), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w1 := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(batchSize*2, batchSize, kernelSize, kernelSize), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w2 := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(batchSize*4, batchSize*2, kernelSize, kernelSize), gorgonia.WithName("w2"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w3 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(batchSize*16, batchSize*8), gorgonia.WithName("w3"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w4 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(batchSize*8, outputSize), gorgonia.WithName("w4"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	return &convnet{
		w0: w0,
		w1: w1,
		w2: w2,
		w3: w3,
		w4: w4,

		d0: 0.2,
		d1: 0.2,
		d2: 0.2,
		d3: 0.55,
	}
}

func (m *convnet) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1, m.w2, m.w3, m.w4}
}

func (m *convnet) Fwd(input *gorgonia.Node) error {
	/*var c0, c1, c2, fc *gorgonia.Node
	var a0, a1, a2, a3 *gorgonia.Node
	var p0, p1, p2 *gorgonia.Node
	var l0, l1, l2, l3 *gorgonia.Node */
	var err error
	var c0, a0, p0, l0 *gorgonia.Node

	// LAYER 0
	// stride = (1, 1) and padding = (1, 1)
	kernelShape := tensor.Shape{m.w0.Shape()[2], m.w0.Shape()[3]}
	c0, err = gorgonia.Conv2d(input, m.w0, kernelShape, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return errors.Wrap(err, "Cannot conv2d Layer0")
	}
	a0, err = gorgonia.Rectify(c0)
	if err != nil {
		return errors.Wrap(err, "Cannot ReLU Layer0")
	}
	p0, err = gorgonia.MaxPool2D(a0, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2})
	if err != nil {
		return errors.Wrap(err, "Cannot MaxPool Layer0")
	}
	//log.Printf("p0 shape %v", p0.Shape())
	l0, err = gorgonia.Dropout(p0, m.d0)
	if err != nil {
		return errors.Wrap(err, "Cannot Dropout Layer0")
	}
	fmt.Println(l0)

	return nil
}
