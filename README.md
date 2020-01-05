# simplest-image-classification-keras

Hi all, <br> This code base is my attempt to give basic but enough detailed tutorial for beginners on image classification using keras in python.

- I used here the basic set of libraries and predict 10 categories of image.

- Implemented a simple CNN Model and used 10 images per category (10x10 = 100 rows of input) to Train and Test the model

- Model performed >70% accuracy which is quite good for this small set of data.

- You can add / update any kind of images and labels and Train the model to meet your needs.

- You can find inline comments to understand the code easily and modify as per your requirement.

<hr>

### You can find about the detailed implementation, usage  and tutorial at my below Medium Post.

```py
    # True : To Re-Train Model with New set of Images and Labels
    # False : To Load existing Trained Model and simply predict
    retrain = False
    if(retrain):
        imageClassification.start(retrain=True)
    else:
        imageClassification.start(retrain=False)
```

![Screenshot](Screenshot.png)