# floor-control experiments

## Evaluating the floor control detection (FCD) model

The model is evaluated using multiple measures, against an annotated dataset, and compared with competitor models.
One of these competitors is the state-of-the-art general turn-taking LSTM model by Gabriel Skantze (Skantze, 2017).
We replicated the model with [keras](https://keras.io/).

## How to use

1. Install dependencies: ``pip install -r requirements.txt``. Using a virtual environment and/or [pip-tools](https://github.com/jazzband/pip-tools) is highly recommended.
1. Download the DUEL dataset (Hough et al. 2016), or at least the German part. Contact the authors to get access.
1. Edit ``settings.py`` with the correct path to the DUEL folder.
1. Optionally, run the ``skantze_prepare.py`` script to collect the features for training the LSTM model. The output of this script is in the ``data`` folder.
1. Optionally, run the ``skantze_train.py`` script to train the LSTM model. The output of this script is the ``model_0.h5`` and ``model_1.h5`` files. Note that it takes about 2-3 days to run on an average 2017 laptop.
1. Run the ``experiments`` notebook.

## Bibliography

- Hough, J., Tian, Y., de Ruiter, L., Betz, S., Kousidis, S., Schlangen, D., and Ginzburg, J. (2016). Duel: A multi-lingual multimodal dialogue corpus for disfluency, exclamations and laughter. In 10th edition of the Language Resources and Evaluation Conference.

- Skantze, G. (2017). Towards a general, continuous model of turn-taking in spoken dialogue using lstm recurrent neural networks. In Proceedings of the 18th Annual SIGdial Meeting on Discourse and Dialogue, pages 220--230.
