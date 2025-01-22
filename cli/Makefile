.PHONY: install clean train test vectorize check train-embedding

clean:
	rm -rf ./out/*

vectorize:
	sxm vectorize \
		--train-file ../cli-data/train.csv

train-model:
	sxm train-model \
		--train-file ../cli-data/train.csv \
		--output-file ./out/model.pt

train-embedding:
	sxm train-embedding \
		--train-file ../cli-data/train.csv


tune:
	sxm tune \
		--train-file ../cli-data/train.csv

test:
	sxm test \
		--test-file ../cli-data/test.csv \
		--model-file ./out/model.pt \
		--output-path ./out/model.test

check:
	sxm check \
		--train-file ../cli-data/train.csv \

all:
	make vectorize
	make train-embedding
	make train
	make test

venv-setup:
	poetry env use 3.12
	poetry env activate

venv-get-path:
	poetry env info --path

