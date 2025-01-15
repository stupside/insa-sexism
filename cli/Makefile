.PHONY: install clean train test vectorize check

clean:
	rm -rf out

vectorize:
	sxm vectorize \
		--train-file ../cli-data/train.csv

train:
	sxm train \
		--train-file ../cli-data/train.csv \
		--output-file ./out/model.pt

test:
	sxm test \
		--test-file ../cli-data/test.csv \
		--model-file ./out/model.pt \
		--output-path ./out/model.test

check:
	sxm check \
		--train-file data/train.csv

all:
	make clean
	make vectorize
	make train
	make test

venv-setup:
	poetry env use 3.12
	poetry env activate

venv-get-path:
	poetry env info --path

