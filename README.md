# CNN-model-with-different-filter-size

### Introduction

Convolutional Neural Networks (CNNs) are widely used in computer vision tasks like image classification. A key component of CNNs is the convolutional layer, which uses filters to extract features from images. The size of these filters plays a critical role in determining the performance, efficiency, and generalization ability of the network. In this tutorial, we will explore the impact of different filter sizes in CNNs, focusing on how they influence feature extraction, computational complexity, and overall model performance.

This tutorial demonstrates how to:
1. Load and preprocess the CIFAR-10 Dataset.
2. Define CNN Architectures.
3. Train model with passing different filter sizes
4. Visualize Results by passing different filter size in CNN Model

---
### Prerequisites
- [![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

- [![Google Colab](https://img.shields.io/badge/Google%20Colab-Data%20Science%20Platform-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)


- Libraries:
  - ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
  - [![matplotlib](https://img.shields.io/badge/matplotlib-008080?style=for-the-badge&logoColor=white)](https://matplotlib.org/)



Install the required libraries using:
```bash
Sklearn
Matplotlib
```

### Running the Notebook
1. Clone the repository:
   ```bash
   git clone https://github.com/akshay-chauhan-1810/CNN-with-different-filter-sizes-Tutorial.git
   ```
2. Navigate to the `notebooks` folder:
   ```bash
   cd CNN-mode-with-different-filter-size-Tutorial/notebooks
   ```
3. Launch Google Colab Notebook:
   ```bash
   Google Colab notebook
   ```
4. Open the `Filter_size_impact_in_CNN.ipynb` file and follow the instructions.
    
## Dataset Description

We performed the experiments with CIFAR10 dataset. This dataset are small and precise datasets having low computational costs. The results drawn from these datasets can be applied to most of the datasets. So, we use this dataset to avoid the computation cost of more extensive datasets. CIFAR10 is the dataset of the 60000 images of these categories plane, car, bird, cat, deer, dog, frog, horse, and ship. This dataset has 10 output classes.

---

## Methods Used

### Step 1: Load and Preprocess Data
- Split the dataset into training and testing sets (50,000 | 10,000) to ensure robust evaluation.
- Normalize pixel values [0,1] Range

### Step 2: Define CNN Architectures
1. Convolutional Layers:
 - Conv2D: 32 filters of specified size (3x3, 5x5, or 7x7).
 - activation='relu': Introduces non-linearity via Rectified Linear Unit.
 - input_shape=(32, 32, 3): Matches CIFAR-10 image dimensions (height, width, channels).
2. Max Pooling Layers:
 - MaxPooling2D((2, 2)): Reduces spatial dimensions by 50% (downsampling).
3. Classifier Head:
 - Flatten(): Converts 2D feature maps to 1D vector for dense layers.
 - Dense(64): Fully connected layer with 64 neurons.
 - Dense(10): Output layer with 10 units (one per class).

### Step 3: Train model 
- Training:
  - Epochs: 10 iterations over the entire dataset.
  - Validation Data: Test set used to evaluate performance after each epoch.

### Setp 4: Results

-	3x3 Filters: Achieved the highest validation accuracy (70%) and were computationally efficient.
-	5x5 Filters: Achieved moderate accuracy (67%) but required more parameters.
-	7x7 Filters: Achieved the lowest accuracy (62%) and overfitted the training data.


---

## Accessibility Considerations

1. **Color-blind-friendly visuals**: Feature importance bar charts use distinguishable shades with text annotations.
2. **Screen reader compatibility**: Key outputs, such as model metrics and feature importance, are described in text.
3. **Concise formatting**: A structured format enhances readability for diverse audiences.
---

## Conclusion

Smaller filter sizes (e.g., 3x3) are recommended for image classification tasks like CIFAR-10 due to their efficiency and effectiveness. Larger filters may be useful in specific scenarios but come with higher computational costs.

---

## References

1.	Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
2.	Zeiler, M.D., & Fergus, R. (2014). Visualizing and Understanding Convolutional Networks. ECCV.
3.	Krizhevsky, A. (2012). ImageNet Classification with Deep CNNs. *NeurIPS*.  
4.	TensorFlow Documentation. https://www.tensorflow.org/  
5.	Szegedy, C., et al. (2015). Going Deeper with Convolutions. CVPR.
6.	TensorFlow Documentation. (2023). Convolutional Neural Networks (CNNs). https://www.tensorflow.org.
7.	https://www.deeplearningbook.org/



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


