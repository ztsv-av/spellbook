# Projects Directory

Directory that contains personal projects and competitions from [Kaggle.com](https://www.kaggle.com/).

Each project directory consists of following folders:

 - `datasets` - contains project datasets and metadata files;
 - `training` - contains trained models' weights and training logs;
 - `best_weights` - contains the best models' weights picked from `training` directory (used for ensemble and prediction);
 - `predictions` - contains predictions for *test* dataset (usually used for score checking on [Kaggle.com](https://www.kaggle.com/) for corresponding competitions).

and `.py` files:

 - `dataPreprocessing.py` - script to transform data into general representation appropriate for fitting (data is saved to `datasets` directory);
 - `prediction.py` - script to prepare and save predictions into `predictions` directory.