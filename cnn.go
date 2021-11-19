package cifarcnn

import (
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
	var c0, c1, c2, fc *gorgonia.Node
	var a0, a1, a2, a3 *gorgonia.Node
	var p0, p1, p2 *gorgonia.Node
	var l0, l1, l2, l3 *gorgonia.Node
	var err error

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
	l0, err = gorgonia.Dropout(p0, m.d0)
	if err != nil {
		return errors.Wrap(err, "Cannot Dropout Layer0")
	}

	// Layer 1
	c1, err = gorgonia.Conv2d(l0, m.w1, kernelShape, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return errors.Wrap(err, "Cannot conv2d Layer1")
	}
	a1, err = gorgonia.Rectify(c1)
	if err != nil {
		return errors.Wrap(err, "Cannot ReLU Layer1")
	}
	p1, err = gorgonia.MaxPool2D(a1, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2})
	if err != nil {
		return errors.Wrap(err, "Cannot MaxPool Layer1")
	}
	l1, err = gorgonia.Dropout(p1, m.d1)
	if err != nil {
		return errors.Wrap(err, "Cannot Dropout Layer1")
	}

	// Layer 2
	c2, err = gorgonia.Conv2d(l1, m.w2, kernelShape, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return errors.Wrap(err, "Cannot conv2d Layer2")
	}
	a2, err = gorgonia.Rectify(c2)
	if err != nil {
		return errors.Wrap(err, "Cannot ReLU Layer2")
	}
	p2, err = gorgonia.MaxPool2D(a2, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2})
	if err != nil {
		return errors.Wrap(err, "Cannot MaxPool Layer2")
	}
	var r2 *gorgonia.Node
	b, c, h, w := p2.Shape()[0], p2.Shape()[1], p2.Shape()[2], p2.Shape()[3]
	r2, err = gorgonia.Reshape(p2, tensor.Shape{b, c * h * w})
	if err != nil {
		return errors.Wrap(err, "Cannot Reshape Layer2")
	}
	l2, err = gorgonia.Dropout(r2, m.d2)
	if err != nil {
		return errors.Wrap(err, "Cannot Dropout Layer2")
	}

	// Layer 3
	fc, err = gorgonia.Mul(l2, m.w3)
	if err != nil {
		return errors.Wrap(err, "Cannot multiplicate l2 and w3")
	}
	a3, err = gorgonia.Rectify(fc)
	if err != nil {
		return errors.Wrap(err, "Cannot activate fc")
	}
	l3, err = gorgonia.Dropout(a3, m.d3)
	if err != nil {
		return errors.Wrap(err, "Cannot Dropout Layer3")
	}

	// output decode
	var out *gorgonia.Node
	out, err = gorgonia.Mul(l3, m.w4)
	if err != nil {
		return errors.Wrap(err, "Cannot multiplicate l3 and w4")
	}
	m.out, err = gorgonia.SoftMax(out)

	return nil
}
