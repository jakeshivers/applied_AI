
.PHONY: venv install split train predict clean

venv:
	python -m venv .venv

install:
	pip install -r requirements.txt

split:
	python scripts/split_dataset.py --input_csv data/gym_feedback.csv --train_csv data/train.csv --valid_csv data/valid.csv --valid_frac 0.2

train:
	python src/train.py

predict:
	python src/predict.py

clean:
	rm -rf artifacts/* __pycache__ .pytest_cache
