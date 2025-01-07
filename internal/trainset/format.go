package trainset

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// format preprocesses a CSV file in memory and returns the fixed content as a byte slice
func format(file *os.File) ([]byte, error) {
	// Create a buffer to hold the processed content
	var buffer bytes.Buffer

	// Use a CSV writer to handle proper quoting and escaping
	writer := csv.NewWriter(&buffer)

	// Regular expression to match Python-style lists
	re := regexp.MustCompile(`\['([^']*)'\]`)

	// Create a CSV reader to read the input file
	scanner := csv.NewReader(file)
	scanner.FieldsPerRecord = -1 // Allow variable number of fields per row

	// Read all rows from the CSV
	rows, err := scanner.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read input file: %w", err)
	}

	// Process each row
	for _, row := range rows {
		for i, field := range row {
			// Fix Python-style lists and quotes in each field
			row[i] = re.ReplaceAllString(field, `["$1"]`)
			row[i] = strings.ReplaceAll(row[i], "'", `"`)
		}
		// Write the processed row to the buffer using the CSV writer
		if err := writer.Write(row); err != nil {
			return nil, fmt.Errorf("failed to write row: %w", err)
		}
	}

	// Flush the writer to ensure all data is written to the buffer
	writer.Flush()
	if err := writer.Error(); err != nil {
		return nil, fmt.Errorf("error flushing writer: %w", err)
	}

	return buffer.Bytes(), nil
}
