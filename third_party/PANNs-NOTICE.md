# PANNs attribution

Adapted code: Qiuqiang Kong and contributors, [audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn), revision `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`, `pytorch/models.py` (Cnn6 and ConvBlock5x5). Code is under the MIT license; see `PANNs-LICENSE`.

Pretrained checkpoint: **Cnn6_mAP=0.343.pth**, from [PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition (Pretrained Models)](https://zenodo.org/records/3987831), distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Published MD5: `e25e26b84585b14c7754c91e48efc9be`.

Citation: Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, Mark D. Plumbley, “PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2020. [Paper](https://arxiv.org/abs/1912.10211).

Modifications: replace the 527-class AudioSet head with a six-class head; freeze convolution blocks 1–3 and all batch-normalization running statistics; fine-tune block 4, pretrained fc1, and the new classifier. Disable prefix dropout and augmentation to support deterministic frozen-prefix caching; use dropout in the trainable tail. `artifacts/sound-panns.pt` contains the resulting modified weights. These weights also reflect non-commercial ESC-50 training data; preserve the project's dataset attribution and use restrictions.
