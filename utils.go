package cifarcnn

import (
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

func pixelWeight(px byte) float64 {
	pixelRange := 255.0
	retVal := float64(px)/pixelRange*0.9 + 0.1
	if retVal == 1.0 {
		return 0.999
	}
	return retVal
}

func Load(typ, loc string) (tensor.Tensor, tensor.Tensor, error) {

	var path string
	switch typ {
	case "train":
		path = "data_batch_1.bin"
	case "test":
		path = "test_batch.bin"
	}

	// Create slices to store our data
	var labels []uint8
	var images []float64

	f, err := os.Open(filepath.Join(loc, path))
	if err != nil {
		return nil, nil, errors.Wrap(err, "Cannot open the file")
	}
	defer f.Close()

	cifar, err := ioutil.ReadAll(f)
	if err != nil {
		return nil, nil, errors.Wrap(err, "Cannot read the file")
	}

	for index, element := range cifar {
		if index%3073 == 0 {
			labels = append(labels, uint8(element))
		} else {
			images = append(images, pixelWeight(element))
		}
	}

	// Transform label slice into the necessary format
	numLabels := 10
	labelsBacking := make([]float64, len(labels)*numLabels, len(labels)*numLabels)
	labelsBacking = labelsBacking[:0]
	for i := 0; i < len(labels); i++ {
		for j := 0; j < numLabels; j++ {
			if j == int(labels[i]) {
				labelsBacking = append(labelsBacking, 1.0)
			} else {
				labelsBacking = append(labelsBacking, 0.0)
			}
		}
	}

	inputs := tensor.New(tensor.WithShape(len(labels), 3, 32, 32), tensor.WithBacking(images))
	targets := tensor.New(tensor.WithShape(len(labels), numLabels), tensor.WithBacking(labelsBacking))

	return inputs, targets, nil
}
