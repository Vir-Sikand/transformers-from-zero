setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	pytest

week0_sanity:
	python -m week0_tensors.run_sanity

week1_overfit:
	python -m week1_nn_basics.run_overfit --epochs 10 --batch_size 64
