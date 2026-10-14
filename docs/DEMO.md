# Demo plan (maximum three minutes)

Run the backend and frontend using the README. Open http://127.0.0.1:5173 before recording. The generated captioned video, if present in `output/`, is a live browser walkthrough without narration; review it before submission or use this script to record your own voice.

## 0:00–0:25 — Problem and model choice

“This is Sound Detective, my environmental audio application. In HW1 it played audio and supported waveform inspection and annotations. For HW2 I added a PyTorch multilayer perceptron that recognizes six categories. I chose a small MLP because the dataset is small and I can inspect the entire training process.”

Show the original player, waveform, and annotations.

## 0:25–1:10 — Live prediction

Scroll to Ask the sound detective. Choose a held-out test recording, play it, and click Identify sound. Show the predicted category, dataset label, and six scores. Then upload a WAV and predict again.

“These predictions are computed from audio by a FastAPI backend. The model receives 168 log-mel and MFCC summary features, then uses 64 ReLU hidden units and six output units. Uploaded WAVs use exactly the same preprocessing as training.”

## 1:10–2:05 — Evidence and limitations

Show Training & evaluation, the loss curves, selected-checkpoint table, and confusion matrix.

“I use 240 recordings from six ESC-50 classes. Folds 1–3 are training, fold 4 is validation, and fold 5 is test. Source recordings do not cross these sets. The model uses cross-entropy and SGD with momentum. Hyperparameters and epoch 8 were chosen on validation data only. Test accuracy is 85.4%, compared with 75% for logistic regression using the same features. Some dog and clock sounds are confused, and temporal averaging loses event order.”

## 2:05–2:35 — Integration and disclosure

Show blind challenge gating or return to the prediction panel.

“Users obtain predictions directly in the web interface. The original library examples are training samples; the separate test examples were held out. Scores are not calibrated certainty, and unfamiliar sounds still get one of the six labels. The repository includes trained weights, reproduction commands, tests, and an AI-assistance disclosure.”

Do not claim interactive training or a live training monitor: neither optional challenge is implemented. Submit the GitHub link and a video of at most three minutes through Canvas.
