package main

import (
	"cifarcnn"
	"log"
	"os"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	f, err := os.Open("data_batch_1.bin")
	if err != nil {
		panic(err)
	}
	inputs, targets, err := cifarcnn.Load(f)
	if err != nil {
		panic(err)
	}

	// Creating a net
	kernelSize := 5
	depth := 3
	classes := 10
	length := 32
	g := gorgonia.NewGraph()
	convnet := cifarcnn.NewCNN(g, length, kernelSize, depth, classes)

	bs := 100
	x := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(bs, depth, length, length), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(bs, classes), gorgonia.WithName("y"))

	// feed forward proccess
	if err = convnet.Fwd(x); err != nil {
		panic(err)
	}

	// Defining cost
	var logprob, losses, cost *gorgonia.Node
	logprob, err = gorgonia.Log(convnet.Out)
	if err != nil {
		panic(err)
	}
	losses, err = gorgonia.HadamardProd(logprob, y)
	if err != nil {
		panic(err)
	}
	cost, err = gorgonia.Sum(losses)
	if err != nil {
		panic(err)
	}
	cost, err = gorgonia.Neg(cost)
	if err != nil {
		panic(err)
	}

	// Track costs
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	// Defining gradients
	_, err = gorgonia.Grad(cost, convnet.Learnables()...)
	if err != nil {
		panic(err)
	}

	tm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(convnet.Learnables()...))
	defer tm.Close()
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(bs)))

	// training
	numExamples := inputs.Shape()[0]
	batches := numExamples / bs
	epochs := 10
	for i := 0; i < epochs; i++ {
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= numExamples {
				break
			}
			if end > numExamples {
				end = numExamples
			}

			var xVal, yVal tensor.Tensor
			if xVal, err = inputs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = targets.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice y")
			}
			if err = xVal.(*tensor.Dense).Reshape(bs, depth, length, length); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err = tm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}

			solver.Step(gorgonia.NodesToValueGrads(convnet.Learnables()))
			tm.Reset()
		}
		log.Printf("Epoch %d | cost %v", i, costVal)

	}

}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }
