package main

import (
	"cifarcnn"
	"fmt"
	"os"

	"github.com/pkg/errors"
)

func main() {
	f, err := os.Open("data_batch_1.bin")
	if err != nil {
		panic(err)

	inputs, targets, err := cifarcnn.Load(f)
	if err != nil {
		panic(err)
	}
	fmt.Println(inputs, targets.Data())
}
