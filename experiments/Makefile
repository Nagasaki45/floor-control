duel:
	python extract_utterances.py
	python extract_fcd_features.py
	python predict_vad.py
	python extract_lstm_features.py

reproduce:
	python predict_fcd.py
	python train_lstm.py
	python predict_lstm.py

all: duel predict
