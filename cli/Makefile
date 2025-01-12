train:
	sxm train --train="../cli-data/train.csv" --test="../cli-data/test.csv" --output="./out/model.pt"

predict:
	sxm predict --model="./out/model.pt" --test="../cli-data/test.csv" --output="./out/model.predictions.csv"

install:
	poetry install

copy-venv:
	poetry env info --path | pbcopy