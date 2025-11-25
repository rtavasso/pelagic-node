  1. uv venv && uv pip install -r requirements.txt -r requirements-training.txt
  2. uv run python generate_dummy_data.py — creates training data
  3. uv run python train_model.py — trains and exports to models/classifier.onnx
  4. Place audio in data/sim_input/
  5. uv run python src/main.py — runs simulation