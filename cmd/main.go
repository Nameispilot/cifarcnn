package main

import (
	"cifarcnn"
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	learning_rate = 0.01
	batchSize     = 1
	imgHeight     = 6
	imgWidth      = 4
	imgChannels   = 1
	classes       = 3
	imgShape      = []int{batchSize, imgChannels, imgHeight, imgWidth}

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

	kernelSize := 3
	cnn := cifarcnn.BuildCNN(g, kernelSize, imgChannels, classes)

	// preparing input node
	input := gorgonia.NewTensor(g, gorgonia.Float64, 4, gorgonia.WithShape(imgShape...), gorgonia.WithName("cnn_input_"))
	err := cnn.Fwd(input)
	if err != nil {
		panic(err)
	}

	// preparing labels node
	target := gorgonia.NewTensor(g, gorgonia.Float64, 1, gorgonia.WithShape(classes), gorgonia.WithName("cnn_labels_"))

	losses, _ := gorgonia.Sub(cnn.Out(), target)
	square, _ := gorgonia.Square(losses)
	cost, _ := gorgonia.Mean(square)

	//defining gradients
	_, err = gorgonia.Grad(cost, cnn.Learnables()...)
	if err != nil {
		panic(err)
	}

	// tracking out value
	var cnnOut gorgonia.Value
	gorgonia.Read(cnn.Out(), &cnnOut)

	// creating a tape machine
	tm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(cnn.Learnables()...))
	defer tm.Close()

	// defining solver
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(batchSize)), gorgonia.WithLearnRate(learning_rate))

	/* training */
	epochs := 500
	for i := 0; i < epochs; i++ {
		// zero
		err := gorgonia.Let(input, zero_image)
		if err != nil {
			panic(err)
		}
		err = gorgonia.Let(target, zero_image_label)
		if err != nil {
			panic(err)
		}

		err = tm.RunAll()
		if err != nil {
			panic(err)
		}

		err = solver.Step(gorgonia.NodesToValueGrads(cnn.Learnables()))
		if err != nil {
			panic(err)
		}
		tm.Reset()

		// one
		err = gorgonia.Let(input, one_image)
		if err != nil {
			panic(err)
		}
		err = gorgonia.Let(target, one_image_label)
		if err != nil {
			panic(err)
		}

		err = tm.RunAll()
		if err != nil {
			panic(err)
		}

		err = solver.Step(gorgonia.NodesToValueGrads(cnn.Learnables()))
		if err != nil {
			panic(err)
		}
		tm.Reset()

		// two
		err = gorgonia.Let(input, two_image)
		if err != nil {
			panic(err)
		}
		err = gorgonia.Let(target, two_image_label)
		if err != nil {
			panic(err)
		}

		err = tm.RunAll()
		if err != nil {
			panic(err)
		}

		err = solver.Step(gorgonia.NodesToValueGrads(cnn.Learnables()))
		if err != nil {
			panic(err)
		}
		tm.Reset()

	}

	/* Testing */
	fmt.Println("------------------RESULTS------------------")
	// zero
	err = gorgonia.Let(input, zero_image)
	if err != nil {
		panic(err)
	}
	err = tm.RunAll()
	if err != nil {
		panic(err)
	}
	tm.Reset()

	fmt.Print("'0'\t[1, 0, 0] ")
	fmt.Printf("%.2f\n", cnnOut.Data())

	// one
	err = gorgonia.Let(input, one_image)
	if err != nil {
		panic(err)
	}
	err = tm.RunAll()
	if err != nil {
		panic(err)
	}
	tm.Reset()

	fmt.Print("'1'\t[0, 1, 0] ")
	fmt.Printf("%.2f\n", cnnOut.Data())

	// two
	err = gorgonia.Let(input, two_image)
	if err != nil {
		panic(err)
	}
	err = tm.RunAll()
	if err != nil {
		panic(err)
	}
	tm.Reset()

	fmt.Print("'2'\t[0, 0, 1] ")
	fmt.Printf("%.2f\n", cnnOut.Data())

}
