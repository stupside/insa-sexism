package main

import (
	"log/slog"
	"os"

	"github.com/gocarina/gocsv"
)

// tweet,annotators,gender_annotators,age_annotators,ethnicities_annotators,study_levels_annotators,countries_annotators,labels_task1,labels_task2,ID
type TrainEntry struct {
	ID                    int      `json:"id"`                      // Unique ID for the tweet
	Tweet                 string   `json:"tweet"`                   // The content of the tweet
	Annotators            []string `json:"annotators"`              // List of annotators
	GenderAnnotators      []string `json:"gender_annotators"`       // Gender of each annotator
	AgeAnnotators         []string `json:"age_annotators"`          // Age groups of annotators
	EthnicitiesAnnotators []string `json:"ethnicities_annotators"`  // Ethnicities of annotators
	StudyLevelsAnnotators []string `json:"study_levels_annotators"` // Educational levels of annotators
	CountriesAnnotators   []string `json:"countries_annotators"`    // Countries of annotators
	LabelsTask1           []string `json:"labels_task1"`            // Labels for Task 1
	LabelsTask2           []string `json:"labels_task2"`            // Labels for Task 2
}

type TrainSet = []TrainEntry

func main() {

	// Open train set file
	file, err := os.OpenFile("./assets/train.csv", os.O_RDWR|os.O_CREATE, os.ModePerm)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	set := TrainSet{}

	// Load train set
	if err := gocsv.UnmarshalFile(file, &set); err != nil {
		panic(err)
	}

	slog.Info("Hello, World!")

	for _, entry := range set {
		slog.Info("Entry", slog.Int("ID", entry.ID))
	}
}
