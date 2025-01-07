package trainset

import (
	"fmt"
	"log/slog"
	"os"

	"github.com/gocarina/gocsv"
)

// tweet,annotators,gender_annotators,age_annotators,ethnicities_annotators,study_levels_annotators,countries_annotators,labels_task1,labels_task2,ID

// tweet: Remember when and why the misogynists in positions of authority changed Women's studies to gender studies at universities all over the country?Remember what Feminists said would happen?I do. https://t.co/TySzlrbiBT,"
// annotators: ['Annotator_461', 'Annotator_462', 'Annotator_463', 'Annotator_464', 'Annotator_465', 'Annotator_466']",
// gender_annotators: "['F', 'F', 'M', 'M', 'M', 'F']",
// age_annotators: "['18-22', '23-45', '18-22', '23-45', '46+', '46+']",
// ethnicities_annotators: "['Asian', 'White or Caucasian', 'White or Caucasian', 'Black or African American', 'White or Caucasian', 'White or Caucasian']",
// study_levels_annotators: "['Bachelor’s degree', 'Bachelor’s degree', 'High school degree or equivalent', 'Bachelor’s degree', 'Bachelor’s degree', 'Bachelor’s degree']",
// countries_annotators: "['United Kingdom', 'Poland', 'Portugal', 'South Africa', 'Greece', 'United Kingdom']",
// labels_task1: "['YES', 'YES', 'YES', 'YES', 'YES', 'YES']",
// labels_task2: "['JUDGEMENTAL', 'JUDGEMENTAL', 'REPORTED', 'REPORTED', 'REPORTED', 'REPORTED']",
// ID: 500
type TrainEntry struct {
	ID                    int      `json:"id" csv:"ID"`                                           // ID of the tweet
	Tweet                 string   `json:"tweet" csv:"tweet"`                                     // Tweet
	Annotators            []string `json:"annotators" csv:"annotators"`                           // Annotators
	GenderAnnotators      []string `json:"gender_annotators" csv:"gender_annotators"`             // Gender of annotators
	AgeAnnotators         []string `json:"age_annotators" csv:"age_annotators"`                   // Age of annotators
	EthnicitiesAnnotators []string `json:"ethnicities_annotators" csv:"ethnicities_annotators"`   // Entnicities of annotators
	StudyLevelsAnnotators []string `json:"study_levels_annotators" csv:"study_levels_annotators"` // Study levels of annotators
	CountriesAnnotators   []string `json:"countries_annotators" csv:"countries_annotators"`       // Countries of annotators
	LabelsTask1           []string `json:"labels_task1" csv:"labels_task1"`                       // Labels for Task 1
	LabelsTask2           []string `json:"labels_task2" csv:"labels_task2"`                       // Labels for Task 2
}

type TrainSet struct {
	Entries []*TrainEntry
}

func NewTrainSet() *TrainSet {
	return &TrainSet{}
}

func (t *TrainSet) Load(file *os.File) error {

	bytes, err := format(file)
	if err != nil {
		return fmt.Errorf("failed to fix train set: %w", err)
	}

	// Load train set
	if err := gocsv.UnmarshalBytes(bytes, &t.Entries); err != nil {
		return fmt.Errorf("failed to load train set: %w", err)
	}

	return nil
}

func (t *TrainSet) Print() {
	for _, entry := range t.Entries {
		slog.Info("entry", slog.Int("ID", entry.ID), slog.String("Tweet", entry.Tweet))
	}
}
