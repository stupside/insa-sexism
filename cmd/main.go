package main

import (
	"os"
	"sexism/internal/trainset"
)

func main() {

	file, err := os.OpenFile("./assets/train.csv", os.O_RDONLY, os.ModePerm)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	trainset := trainset.NewTrainSet()

	if err := trainset.Load(file); err != nil {
		panic(err)
	}

	trainset.Print()
}
