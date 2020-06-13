# GAN-online-and-offline-augmentation-to-improve-skin-cancer-classification
This contains the codes and the report of a part of my final year undergraduate project "Applications of Generative Adversarial Network". Here, the dataset used is the public "HAM10000" dataset which contains images of skin cancers. First, a Convoloutional Neural Network (CNN) Model was trained to classify the diseases(the top 4 diseases only considered). Due to the high data imbalance, the model overfits on the frequent class. To combat this, data augmentation is used by augmenting the dataset with artificially generated images generated by a Deep Convolutional Generative Adversarial Network (DCGAN). Here, both approaches of online and offline GAN augmentation were used and compared. The approach of online GAN augmentation was part of my summer research project at Indian Statistical Institute, Kolkata in 2018 where it was performed on the NIH Chest X-Ray dataset which has the similar property of data imbalance. The published book chapter for the same is given in reference.

# Reference
Bhattacharya D., Banerjee S., Bhattacharya S., Uma Shankar B., Mitra S. (2020) GAN-Based Novel Approach for Data Augmentation with Improved Disease Classification. In: Verma O., Roy S., Pandey S., Mittal M. (eds) Advancement of Machine Intelligence in Interactive Medical Image Analysis. Algorithms for Intelligent Systems. Springer, Singapore

