package cifarcnn

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Convnet struct {
	Name   string
	Layers []*Layer
	out    *gorgonia.Node
}

func (net *Convnet) Out() *gorgonia.Node {
	return net.out
}

func CNN(Layers ...*Layer) *Convnet {
	return &Convnet{
		Name:   "convnet",
		Layers: Layers,
	}
}

func (net *Convnet) Learnables() gorgonia.Nodes {
	learnables := make(gorgonia.Nodes, 0, 2*len(net.Layers))
	for _, l := range net.Layers {
		if l != nil {
			if l.Weights != nil {
				learnables = append(learnables, l.Weights)
			}
			if l.Bias != nil {
				learnables = append(learnables, l.Bias)
			}
		}
	}
	return learnables
}

func BuildCNN(g *gorgonia.ExprGraph, kernelSize, channels, labels int) *Convnet {
	shp0 := tensor.Shape{5, channels, kernelSize, kernelSize}
	w0 := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(shp0...), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w1 := gorgonia.NewTensor(g, gorgonia.Float64, 2, gorgonia.WithShape(labels, 10), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	convnet := CNN(
		[]*Layer{
			{
				Type:       LayerConvolutional,
				Weights:    w0,
				Bias:       nil,
				Activation: Rectify,
				Extra: &Extra{
					KernelHeight: kernelSize,
					KernelWidth:  kernelSize,
					Padding:      []int{1, 1},
					Stride:       []int{1, 1},
					Dilation:     []int{1, 1},
				},
			},
			{
				Type:       LayerDropout,
				Activation: NoActivation,
				Extra: &Extra{
					Probability: 0.2,
				},
			},
			{
				Type:       LayerMaxpool,
				Activation: NoActivation,
				Extra: &Extra{
					KernelHeight: kernelSize,
					KernelWidth:  kernelSize,
					Padding:      []int{0, 0},
					Stride:       []int{2, 2},
				},
			},
			{
				Type:       LayerFlatten,
				Activation: NoActivation,
			},
			{
				Type:       LayerLinear,
				Weights:    w1,
				Bias:       nil,
				Activation: SoftMax,
			},
		}...,
	)
	return convnet

}

func (net *Convnet) Fwd(input *gorgonia.Node) error {
	firstLayerToActivate, err := net.Layers[0].Fwd(input)
	if err != nil {
		return errors.Wrap(err, "Can't feedforward Layer 0")
	}
	firstLayerActivated, err := net.Layers[0].Activation(firstLayerToActivate)
	if err != nil {
		return errors.Wrap(err, "Can't apply activation function to Layer 0")
	}
	lastActivatedLayer := firstLayerActivated

	for i := 1; i < len(net.Layers); i++ {
		layerNonActivated, err := net.Layers[i].Fwd(lastActivatedLayer)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("Can't feedforward Layer #%d", i))
		}
		layerActivated, err := net.Layers[i].Activation(layerNonActivated)
		if err != nil {
			return errors.Wrap(err, fmt.Sprintf("Can't apply activation function to Layer #%d", i))
		}
		net.Layers[i].output = layerActivated
		lastActivatedLayer = layerActivated
		if i == len(net.Layers)-1 {
			net.out = layerActivated
		}
	}

	return nil
}
