package cifarcnn

import "gorgonia.org/gorgonia"

type ActivationFunc func(a *gorgonia.Node) (*gorgonia.Node, error)

func NoActivation(a *gorgonia.Node) (*gorgonia.Node, error) { return a, nil }
func Rectify(a *gorgonia.Node) (*gorgonia.Node, error)      { return gorgonia.Rectify(a) }
func SoftMax(a *gorgonia.Node) (*gorgonia.Node, error)      { return gorgonia.SoftMax(a) }
