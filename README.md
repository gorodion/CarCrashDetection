# CarCrashDetection
*Please, note that our models require a lot of memory, so running on Google Colab is a must*
## Our idea
The key idea of our solution is to use [Temporal Convolutional Networks (TCN)](https://link.springer.com/content/pdf/10.1007/978-3-319-49409-8_7.pdf). We extract low-level features using a ResNet encoder and then pass these features to our TCN. 
We inspect video-frames using a small window (size: 300 frames / 10s, since most accidents did not last longer) and try detecting car accidents in this frame. If an accident was detected, message about it is logged and the accident is then saved. In order to make our approach time-efficient we halt execution after an accident is found.

## Instruction
In order to run our model, please, do the following:
* Install the requirements `python -m pip install -r requirements.txt`
* Download pretrained models from the [folder](https://drive.google.com/drive/folders/1sZV0zNi0Av7DVZf88DUgmu4LKVdimdHD?usp=sharing) (clickable)
* Put the pretrained models to the `models` folder or just write path to the models in `config.py`
* If you want to run our program, run the following command in cmd: `python main.py --path=PATH_TO_VIDEOS`, where `PATH_TO_VIDEOS` is the folder with the test sample of videos
* Results can be seen in the `predictions.csv` file. There 0 means "No accident detected" and 1 means "An accedent detected"

## Repository structure
* parsing/ - scripts for parsing and resulting files
* DTPClassifier.ipynb - training model based on ResNet18 and TCN to detect an accident on series of frames
* accident.py - scripts to run accident model
* detector.py - scripts to run detector model
* accident_logger.py - script to save video with a car accident

## Possible improvements
* **Use attention**. Car accidents occure in different regions of the screen, so "*paying attention*" to these regions could significantly improve the performance.
* **Use Detection / Tracking**. A good idea is to get car boxes using a tracking algorithm and recognize accidents when a car is in focus.
* **Extract extra features with segmentation**. We can extract more features: information about traffic lights, road, etc using a segmentation model. Adding these feature maps to video can give the model an understanding of the "global scene". 
* **Use additional markers**. An accident can be recognized using several markers: emergency lights, crowding, etc.
