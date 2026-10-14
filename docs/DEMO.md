# Final demo — includes both optional challenges

Use `output/sound-detective-hw2-extra-credit.mp4` for the final captioned walkthrough when available. It supersedes the earlier MLP-only and two-model videos. The video has English captions and no narration. Review before submission; maximum three minutes.

1. Introduce the existing HW1 audio viewer and the two published models: MLP from scratch and AudioSet-pretrained PANNs with partial fine-tuning.
2. In **Interactive training lab**, change learning rate, epoch count and hidden units, then click Start training. Point out live progress and train/validation loss and accuracy. This is actual background training, not replayed metrics.
3. Show the completed table: train, validation and test loss/accuracy/macro F1. Explain that the lowest validation-loss checkpoint is restored before test evaluation.
4. Select **Interactive MLP · latest completed run** and predict a held-out recording; show the result and probability bars.
5. Show the published three-model comparison: logistic regression 75.0%, MLP 85.4%, fine-tuned PANNs 95.8%. These fixed experiments stay separate from interactive runs.
6. Mention the executed notebooks, saved weights, AI assistance disclosure and limitations. The test split has 48 clips; score estimates are limited, and PANNs also had extra pretraining data.

Both optional challenges are now implemented. Parameter controls: learning rate, epochs, hidden units and dropout. Monitor: live train/validation curves, progress, final three-split metrics, and prediction results. Stop/cancellation, refresh recovery, and worker failures are also handled. Submit the GitHub branch/merged repository link and final video through Canvas.
