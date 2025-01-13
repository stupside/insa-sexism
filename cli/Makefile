train:
	sxm train --train="../cli-data/train.csv" --test="../cli-data/test.csv" --output="./out/model.pt"

predict:
	sxm predict --model="./out/model.pt" --test="../cli-data/test.csv" --output="./out/model.predictions.csv"

check-balance:
	sxm check-balance --train="../cli-data/train.csv"

venv-setup:
	poetry env use 3.12
	poetry env activate

venv-get-path:
	poetry env info --path

