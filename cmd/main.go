package main

import (
	"cifarcnn"
	"fmt"
)

func main() {

	inputs, targets, err := cifarcnn.Load("train", "/home/bublik/cifar_cnn/cifar-10/")
	if err != nil {
		panic(err)
	}
	fmt.Println(inputs, targets.Data())
}
