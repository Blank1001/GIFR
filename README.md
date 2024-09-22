# GIFR
## Dataset Download

We used **MNER-MI** and **MNER-MI-Plus** as datasets.

The download link is as follow: https://pan.baidu.com/s/1IcJ74WoxfUyO1f1pM9k9Rg?pwd=465s

After downloading, the data needs to be extracted and placed in the directory shown below:

```bash
Dataset
|-- text
|    |-- MNER-MI_train.txt   # The training set of MNER-MI.
|    |-- MNER-MI_val.txt     # The validation set of MNER-MI.
|    |-- MNER-MI_test.txt    # The test set of MNER-MI.
|    |-- MNER-UNI_train.txt  # The training set of MNER-MI-Plus.
|    |-- MNER-UNI_val.txt    # The validation set of MNER-MI-Plus.
|    |-- MNER-UNI_test.txt   # The test set of MNER-MI-Plus.
|    |-- data_UNI.txt        # The set of D_model.
|    |-- preference2.txt     # The set of D_dis.
|-- new_images.zip -> new_images (folder)
|-- twitter2017_images.zip -> twitter2017_images (folder)
```

The shared cloud files include a folder named `text`, and two zip files named `new_images.zip` and `twitter2017_images.zip`.

The `new_images.zip` contains all the images from the MNER-MI dataset, while the `twitter2017_images.zip` contains all the images from the Twitter-2017 dataset.

## Required pre-trained models
In our paper, we use `BERT` and `VIT` as text encoder and image encoder, respectively.

For the code implementation, we utilized the models and weights provided by Hugging Face. 
Specifically, in lines 51 and 54 of the `run.py`, we downloaded the corresponding models (bert-base-uncased and ViTB-16) from the Hugging Face model repository to the local folders. 
The respective Hugging Face links are: https://huggingface.co/google-bert/bert-base-uncased and https://huggingface.co/google/vit-base-patch16-224.

## Running
After you have prepared the required enviroment, data, and models, you can run `python Discriminator.py` to train Relevance-based Image Discriminator and generate a `reward_data.txt` file, in which the saved data is the relevance score of each image, used for subsequent training. Then, you can use the `python run.py` command to train the MNER model.

If running with the `python run.py --dataset MI`, the training and testing will be performed on the **MNER-MI** dataset.
If running with the `python run.py --dataset UNI`, the training and testing will be performed on the **MNER-MI-Plus** dataset.
