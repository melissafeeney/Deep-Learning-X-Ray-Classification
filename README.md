# **Deep-Learning-X-Ray-Classification**

## ***Can Deep Learning be used to correctly identify X-Rays of patients with COVID-19 and viral pneumonia?***
<br>

## **1. Dataset Discussion**
The dataset contains 1,200 COVID-19 positive chest X-Ray images, 1,341 normal chest X-Ray images, and 1,334 viral pneumonia chest X-Ray images. All of these images are compiled from different medical and publically available resources. The objective of this dataset is to serve as a repository from which researchers can perform useful and impactful work on COVID-19. Making this dataset publically available allows for people not usually privy to this type of information a chance to apply their image classification skillsets toward generating a solution for a worldwide problem.

Building a predictive model using this data is certainly useful. Such a model would be able to determine, from a single chest X-Ray image, if the image represented a case of COVID-19, viral pneumonia, or if it was normal. This model would not infringe upon medical professionals' duties in diagnosing patients, but rather serve as an aid in scaling diagnostic abilities. For example, a physician might need multiple chest X-Rays and potentially additional testing to confirm a case of COVID-19. This process could be inefficient, time consuming, and naturally prone to human error. This creates an excellent opportunity for machine learning to be used to process single chest X-Rays efficiently, at scale, and without human error. A model trained to recognize the differences in chest X-Rays between affirmed cases of COVID-19 from viral pneumonia and normal condition could allow physicians to allot their time more productively in treating patients. A model could also detect patterns that differentiate COVID-19 chest X-Rays from those without the condition that the human eye could not detect.

A successful model in this application would benefit patients in receiving accurate diagnoses, physicians in allowing them to spend more time treating patients, and society at large in being able to quickly identify and isolate those with positive cases of COVID-19.
<br>
<br>

## **2. Four different Convolutional Neural Network (CNN) models were tested for this classification task, all varying in architectures**
 The first 3 models are all variations of a similarly structured CNN model- featuring 4 sets of a convolutional layer followed by a max pooling layer, a flatten layer, and then 2 fully connected layers, and then the output layer using softmax activation. In the convolutional layers, the kernel size is 2, the filter sizes are 128, 64, 32 and 32 (respectively, for each of the 4 Conv2D layers), and the padding is set to “same” to ensure that the output size is the same as the input size. The max pooling layer that follows each convolutional layer uses a pool size of 2. The activation function used on the internal layers is relu, and the activation function used on the output layer is softmax. After the convolutional layers are flattened, the two internal densely connected layers have 100 and 50 hidden nodes, respectively. The output layer has 3 nodes to represent the 3 classification groups that the model intends to predict- COVID-19, viral pneumonia, and normal lungs. The fourth model is a transfer learning approach using CheXNet, which is described in more detail below.

All four models use categorical cross entropy as the loss function and adam as the optimizer. All models started out with a learning rate of 0.001, which was reduced upon reaching a plateau, decreasing by a factor of 0.1 after 3 epochs if no improvement in validation accuracy was detected. There is no lower bound imposed upon the learning rate, which became quite small for some of the models. All models were trained on 50 epochs with default batch sizes. 

- **Model 1**- CNN as described above, no batch normalization and no dropout layers
  - In this model, no batch normalization or dropout layers were added, in order to set a baseline upon which subsequent models with batch normalization and dropout layers could be compared. 
 
- **Model 2**- CNN as described above, with batch normalization but no dropout layers
  - In this model, batch normalization was added after every max pooling layer. Batch normalization works through batches of the data to normalize the output of each layer. Along with stabilizing the learning process and potentially reducing the number of epochs needed during training, this can also help to prevent overfitting.
 
- **Model 3**- CNN as described above, with batch normalization and dropout layers
  - In this model, in addition to the same batch normalization layers as in the previously discussed model, two dropout layers were added. Dropout layers are used to prevent overfitting and improve generalization abilities of models, by randomly dropping nodes during the training process. The first dropout layer was added after the first batch normalization layer, and the second was added after the last batch normalization layer. The first dropout layer has a dropout rate of 0.8, and the second dropout layer has a dropout rate of 0.2. 
 
- **Model 4**- CNN with Transfer Learning, applied CheXNet
  - This model was an application of transfer learning and one that I was really excited to try, using an algorithm called CheXNet. After learning in class that models that can have particular abilities with certain kinds of data based on their training, I decided to research if there were any models developed for classifying X-Ray images. That is how I discovered CheXNet, which was built to classify chest X-Rays in 2017! CheXNet is built on top of DenseNet121 and is specifically created for X-Ray image data. I thought it would be interesting to compare the classification abilities of a model that was trained on X-Ray data with other kinds of CNNs. CheXNet was originally trained to detect 14 different medical conditions from X-Rays- including detection of masses, modules, and pneumonia. The model was trained on a dataset of over 112,000 frontal-view chest X-Rays, annotated by radiologists. CheXNet performed exceedingly well in identifying all of the conditions from the chest X-Rays, even outperforming the average radiologist performance using an F1 score- affirming its state of the art status.
  - In this transfer learning process, pre-trained weights from CheXNet are loaded from the aforementioned GitHub link. However, the final layer is altered to fit this classification task, which has 3 potential classes instead of the 14 for which CheXNet was originally trained.
  - DenseNet121 contains 121 layers with trainable weights, and something that makes it unique is that each layer is connected to all the layers that are deeper in the network, hence the name “dense net”. The layers are grouped into 4 groups of dense blocks, each of which contains convolutional, pooling, batch normalization and non-linear activation layers. 
<br>

## **3. X-Ray Visualizations**
**X-Ray Images of Patient with COVID-19**
<br>
![covid_xray](https://user-images.githubusercontent.com/31778500/191571341-6bdad6f0-a8b6-488b-a1de-12739862c7bf.png)

**X-Ray Images of Healthy/Normal Patient**
<br>
![normal_xray](https://user-images.githubusercontent.com/31778500/191571351-5368e499-3e11-4e64-bf04-ffdadd19d849.png)

**X-Ray Images of Patient with Viral Pneumonia**
<br>
![pneumonia_xray](https://user-images.githubusercontent.com/31778500/191571410-11b78498-719b-42f0-bf17-9dba94984ff4.png)


## **4. Model Performance Evaluations**
In terms of model performance, model 4 using CheXNet was best. On the test data, it earned an accuracy score of 98.71%, and a loss of 0.067. I was impressed to see that the model performed really well across all metrics- including F1 score, recall, and precision. I believe that its strong performance can be attributed to its familiarity with chest X-Ray image data. Although it was not previously trained to detect COVID-19, since it was created before the pandemic, having been trained to detect the visual differences that distinguish the X-Ray of a healthy patient from an unhealthy one probably served the model well in its new task to detect COVID-19. In addition, the deep depth of the model coupled with its regularization most likely contributed to its strong performance.

That being said, the other models did not perform poorly. The first model (without batch normalization or dropout layers) yielded a test set accuracy of 97.81% and a loss of 0.071. Similarly, the second model (which included batch normalization but no dropout layers) yielded a test set accuracy of 97.17% and a loss of 0.093. Model 3 (with batch normalization and dropout layers) yielded a test set accuracy of 71.17% and a loss of 2.377.

More detailed results in the classification reports can be seen below:
<br>
<img width="472" alt="Screen Shot 2022-09-21 at 12 42 52 PM" src="https://user-images.githubusercontent.com/31778500/191569665-5975fd87-47e1-4fbc-9482-18990763e55e.png">
<img width="477" alt="Screen Shot 2022-09-21 at 12 43 00 PM" src="https://user-images.githubusercontent.com/31778500/191569680-8fa7dc29-23b2-49e0-82b8-d4b2dc84014d.png">
<br>
<br>

## **5. Citations**
*Citation of paper providing original dataset:*

Chowdhury, M.E.H., Rahman, T. Khandakar, A., Mazhar, R., Kadir, M.A., Mahbub, Z.B., Islam, K.R., Khan, M.S., Iqbal, A., Al-Emadi, N.Reaz. M.B.I. (2020). Can AI help in screening Viral and COVID-19 pneumonia? *IEEE Access 2020*. https://doi.org/10.48550/arXiv.2003.13145

*Citation for the CheXNet Paper:*

Rajpurkar, P., Irvin, J., Zhu, K., Yang, B., Mehta, H., Duan, T., Ding, D., Bagul, A., Langlotz, C., Shpanskaya, K., Lungren, M., Ng, A. (2017). 
CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. https://doi.org/10.48550/arXiv.1711.05225
