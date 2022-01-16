# CarCrashDetection
*Please, note that our models require a lot of memory, so running on Google Colab is a must*

In order to run our model, please, do the following:
* Install the requirements `python -m pip install -r requirements.txt`
* Download pretrained models from the [folder](https://drive.google.com/drive/folders/1sZV0zNi0Av7DVZf88DUgmu4LKVdimdHD?usp=sharing) (clickable)
* Put the pretrained models to the `models` folder or just write path to the models in `config.py`
* If you want to run our program, run the following command in cmd: `python main.py --path=PATH_TO_VIDEOS`, where `PATH_TO_VIDEOS` is the folder with the test sample of videos
* Results can be seen in the `predictions.csv` file. There 0 means "No accident detected" and 1 means "An accedent detected"

## Repository structure
* parsing/ - scripts for parsing and resulting files
* DTPClassifier.ipynb - training model based on ResNet18 and TCN to detect an accident on series of frames
* CarsClassifier.ipynb - training model to differ ordinary and emergency cars
* accident.py - scripts to run accident model
* detector.py - scripts to run detector model
* CarsClassifier.py - scripts to run car classification model (emergency or not)
