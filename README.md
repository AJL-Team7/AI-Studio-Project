# GitHub Kaggle Project AJL Team 7 ReadMe

| Chinonso Morsindi | @ChinonsoM | Selected and finetuned ideal pre-trained models for project, conducted data exploration and visualization |

| Tanisha Jain | @tanisha.jainn | Loading datasets and converted the images into Tensor Flows |

| Shiven Umeshbhai | @AliceDoe | Built CNN model, performed data augmentation |

| Grace Cao | @GuangjingCao | Built and fine-tuned ResNet-based transfer learning models for image classification using TensorFlow; Applied data augmentation and layer freezing to improve generalization |

| Sophia Huang | @2018Shmn | Implemented explainability tools |

| Lam Nguyen | @lamphuong02 | Data exploration, Examine References |

---

## **🎯 Project Highlights**

* Built a CNN and ResNet model using transfer learning to solve image classification through deep learning
* Achieved an F1 score of 0.22 and a ranking of 66 on the final Kaggle Leaderboard
* Used  using CNN and ResNet to classify images to interpret model decisions
* Implemented data augmentation and pixel normalization to optimize results within compute constraints

🔗 [Equitable AI for Dermatology | Kaggle Competition Page](https://www.kaggle.com/competitions/bttai-ajl-2025/overview)

---

## **👩🏽‍💻 Setup & Execution**

* Open Google Colab in your browser and open our notebook
* Import Libraries of Pandas, Numpy, TensorFlow and Sklearn 
* Dataset is stored in a shared Google Drive Folder, includes test data and train data
* The process of how to run our notebook is:
Mount Drive
Install/verify dependencies
Load data from Drive
Run the training notebook
Evaluate and predict

---

## **🏗️ Project Overview**

We trained a model to recognize 21 types of skin conditions using deep learning. We used popular models like EfficientNetB0 and ResNet50, and applied transfer learning so that our model could learn faster and better. We also added things like image rotation and flipping to make our dataset more diverse and help the model generalize. Our best model achieved an F1 score of 0.22, and even though the competition scores were lower, we learned a lot from the experience.
---

## **📊 Data Exploration**

To explore the data, we used various bar graphs, pie charts, and heatmaps to visualize different distributions and relationships across the datasets provded by the AJL Kaggle competition.
* No additional sources were used and we will assume that all skin conditions in the provided dataset have been labeled correctly
* Challenges faced when working with this image dataset is that we are using real-life photos that likely contain some noise or distortion across each sample which could reduce our model accuracy

---

## **🧠 Model Development**
For this project, we utilized **EfficientNetB0**, a state-of-the-art pretrained convolutional neural network (CNN), as the foundation of our model. EfficientNetB0 was chosen for its efficiency and high accuracy across various image classification tasks. The base model was fine-tuned to classify 21 skin conditions from the Fitzpatrick17k dataset.

### **Feature Selection and Hyperparameter Tuning Strategies**
The following strategies were implemented to optimize model performance:
- **Data Augmentation**: To address class imbalances and improve generalization, we applied transformations such as rotation (up to 20°), width/height shifts (20%), horizontal flipping, and pixel normalization ($$ \text{rescale}=1/255 $$).
- **Transfer Learning**: The pretrained EfficientNetB0 weights (trained on ImageNet) were leveraged by freezing its layers initially. Custom layers were added on top, including:
  - A **Global Average Pooling layer** to reduce spatial dimensions.
  - A fully connected **Dense layer** with 128 neurons and ReLU activation.
  - A **Dropout layer** (rate = 0.5) to mitigate overfitting.
  - An output layer with a softmax activation function for the 21-class classification task.
- **Fine-Tuning**: After initial training, the base model layers were unfrozen, and the entire model was fine-tuned with a reduced learning rate ($$10^{-5}$$).

### **Training Setup**
- **Data Split**: The dataset was split into training (80%) and validation (20%) subsets. Generators were created using TensorFlow's `ImageDataGenerator` for efficient batch-wise processing.
- **Batch Size & Input Size**: Images were resized to $$224 \times 224$$ pixels, and a batch size of 32 was used.
- **Loss Function & Optimizer**:
  - Loss: Sparse categorical crossentropy
  - Optimizer: Adam optimizer with an initial learning rate of $$10^{-4}$$.
- **Early Stopping & Checkpoints**:
  - Early stopping monitored validation accuracy with a patience of 10 epochs.
  - Model checkpoints saved the best-performing model during training.

### **Training Process**
1. The initial phase trained only the custom layers on top of the frozen EfficientNetB0 base for up to 50 epochs.
2. In the fine-tuning phase, all layers were unfrozen, and training continued for an additional 20 epochs with a lower learning rate ($$10^{-5}$$).

### **Evaluation Metric**
The primary evaluation metric was the weighted average F1 score, which accounts for class imbalances in the dataset. Validation accuracy was also monitored during training to prevent overfitting.


---

## **📈 Results & Key Findings** 
* Performance metrics in Kaggle Leaderboard score in Private is 0.05395
* 
* Performance metrics in Kaggle Leaderboard score in Public is 0.04708
* 
* The best performance metrics that we got is 0.22


## **🖼️ Impact Narrative**

---

## **🚀 Next Steps & Future Improvements** 
Our implementation utilizes ResNet50 and EfficientNet-B0, but there may be specialized models for medical imaging that may perform better. 
With more time, we would find the optimal hyperparameters and perform more robust k-fold cross-validation. With more resources, we would seek feedback from dermatologists to understand potential clinical application of our model.
Some techniques we could further explore are image augmentation techniques and incorporating metadata about the patient

---

## **📄 References & Additional Resources**

---
