package cifarcnn

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Layer struct {
	Weights    *gorgonia.Node
	Bias       *gorgonia.Node
	Activation ActivationFunc
	Type       LayerType

	output *gorgonia.Node
	Extra  *Extra
}

type LayerType uint16

const (
	LayerLinear = LayerType(iota)
	LayerFlatten
	LayerConvolutional
	LayerMaxpool
	LayerDropout
)

type Extra struct {
	KernelHeight int
	KernelWidth  int
	Padding      []int
	Stride       []int
	Dilation     []int
	Probability  float64
}

func (layer *Layer) Fwd(input *gorgonia.Node) (*gorgonia.Node, error) {
	layerNonActivated := &gorgonia.Node{}
	var err error

	switch layer.Type {
	case LayerConvolutional:
		if layer.Extra == nil {
			return nil, fmt.Errorf("There are no extra options for layer")
		}
		layerNonActivated, err = gorgonia.Conv2d(input, layer.Weights, tensor.Shape{layer.Extra.KernelHeight, layer.Extra.KernelWidth}, layer.Extra.Padding, layer.Extra.Stride, layer.Extra.Dilation)
		if err != nil {
			return nil, errors.Wrap(err, "Can't convolve[2D] input by kernel of layer")
		}
	case LayerDropout:
		layerNonActivated, err = gorgonia.Dropout(input, layer.Extra.Probability)
		if err != nil {
			return nil, errors.Wrap(err, "Can't dilute input of layer")
		}
	case LayerMaxpool:
		if layer.Extra == nil {
			return nil, fmt.Errorf("There are no extra options for layer")
		}
		layerNonActivated, err = gorgonia.MaxPool2D(input, tensor.Shape{layer.Extra.KernelHeight, layer.Extra.KernelWidth}, layer.Extra.Padding, layer.Extra.Stride)
		if err != nil {
			return nil, errors.Wrap(err, "Can't maxpool[2D] input by kernel of layer")
		}
	case LayerFlatten:
		layerNonActivated, err = gorgonia.Reshape(input, tensor.Shape{1, input.Shape().TotalSize()})
		if err != nil {
			return nil, errors.Wrap(err, "Can't flatten input of layer")
		}
	case LayerLinear:
		tOp, err := gorgonia.Transpose(layer.Weights)

		layerNonActivated, err = gorgonia.BatchedMatMul(input, tOp)
		if err != nil {
			return nil, errors.Wrap(err, "Can't multiplicate input and weights")
		}
	}

	layer.output = layerNonActivated
	return layer.output, nil
}
