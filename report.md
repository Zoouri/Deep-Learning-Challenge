# *Analysis Report*

## Overview of the Analysis

The objective of this analysis was to assist the non-profit organization Alphabet Soup in developing an algorithm to predict whether applicants for funding will be successful. Leveraging my expertise in Machine Learning and Neural Networks, I was tasked with creating a binary classifier capable of making such predictions, using the provided dataset.

## Results



##  *Data Preprocessing*

The variable that was considered a target for my model is column `IS_SUCCESSFUL`. 

        |FEATURES                | TARGET           | 
        |------------------------|------------------|
        |`NAME `                 | `IS_SUCCESSFUL`  | 
        |`APPLICATION_TYPE`      |                  | 
        |'AFFILICATION           |                  | 
        |`CLASSIFICATON`         |                  |
        |`USE_CASE`              |                  |
        |`ORGANIZATION`          |                  |
        |`STATUS`                |                  |
        |`INCOME_AMT`            |                  |
        |`SPECIAL_CONSIDERATIONS`|                  | 
        |`ASK_AMT`               |                  |          


During my Optimization I decided to further drop several more irrelevant columns such as: `SPECIAL_CONSIDERATIONS` and `STATUS`   which had no significant impact of the dataset.


Step by step: 

- Loaded the Data: I read charity_data.csv into a Pandas DataFrame to start the analysis.

- Defined Target and Features: I identified the target variable for my model and the feature variables.

- Cleaned the Data: I removed the EIN and NAME columns to focus on relevant information.

- Analyzed Unique Values: I checked the number of unique values for each column. For those with more than 10 unique values, I calculated the count of data points for each unique value.

- Binned Rare Categories: I combined rare categorical values into a new "Other" category, then confirmed that the binning was successful.

- Encoded Categorical Variables: I used pd.get_dummies() to convert categorical variables into one-hot encoded format.

- Split Data for Training and Testing: I split the data into features (X) and target (y), then used train_test_split() to create training and testing datasets.

- Standardized the Data: I used StandardScaler() to fit and transform the training and testing features to standardize the datasets.

##  *Compile, Train, and Evaluate the Model*
I aspired to achieve level of accuracy above 75% however, after multiple attempts I only came as close as 73% at its maximum value. 

My intend was to explore and try different variations to the model and analyse the impact of them. All together I have manipulated the dataset in 4 times:

#### *First Attempt*
I used a fewer number of nodes and only 2 hidden layers to examine how much the data would differ if I further decide to increase these numbers. I achieved accuracy od 73% and data loss of 55%. 

####  *Second Attempt* 
I increased the number of nodes and kept the same amount of hidden layers as well as activation function. It became evident that despite increasing the nodes my results were comperable with accuracy levelled at 73% and data loss at 55%. 

####  *Third Attempt*
During this attempt I decided to increase the number of nodes, add additional layer and change activation function. Activation used during this optimization was relu, tanh, sigmoid and sigmoid for the outer layer. Despite increased numbers, layers and activation methods my models accuracy achieved again 73% of accuracy and 55% data loss. 

#### *Fourth Attempt*
In the last attempt I have decreased the number of nodes but increased number of epochs to test if it would have any significance. Unfortunately, model's accuracy returned 72%  with 55% data loss which has proven no significance in this instance. 

Overall, after multiple attepmts I failed to achieve accuracy of above 75%. 

Step by step: 
- Designed the Neural Network: I used TensorFlow and Keras to design a neural network for a binary classification model to predict whether an Alphabet Soup-funded organization would be successful, based on the features in the dataset.

- Configured the Input Layer: I determined the number of input features and set the appropriate number of neurons and layers in the model.

- Created Hidden Layers: I designed the first hidden layer with a suitable activation function. If needed, I added a second hidden layer with another activation function.

- Set Up the Output Layer: I configured the output layer with an activation function for binary classification.

- Compiled and Trained the Model: I compiled the model, chose an optimizer and loss function, and trained it with the training dataset.

- Added Callbacks: I created a callback to save the model's weights every five epochs, ensuring that I could recover the model's state at key points.

- Evaluated the Model: After training, I evaluated the model on the test dataset to calculate loss and accuracy, confirming its performance.

- Saved the Model and Results: I saved and exported the trained model to an HDF5 file for future use, naming the file appropriately for easy identification.

## *Optimization*

- New Jupyter Notebook: I created a new Jupyter Notebook for optimization, named "AlphabetSoupCharity_Optimisation.ipynb."

- Import Dependencies: I imported the required libraries and read the charity_data.csv into a Pandas DataFrame.

- Data Preprocessing.

Dropping irrelevant columns to avoid confusion in the model (`SPECIAL_CONSIDERATIONS` and `STATUS`)

Creating more bins for rare occurrences.

Increasing or decreasing the number of values per bin to ensure consistent data representation.

- Model Optimization: I optimized the model using various techniques to aim for an accuracy higher than 75%, including:

Adding more neurons to the hidden layers.

Adding more hidden layers to improve model capacity.

Using different activation functions in hidden layers for better non-linear representation.

Changing the number of training epochs to refine the model.

- Training the Optimized Model: I designed and compiled the optimized neural network, then trained it to evaluate its accuracy. Adjusted training parameters, such as epochs and batch size, to enhance model performance.

- Model Evaluation.

- Export Results: I saved and exported the optimized model to an HDF5 file, naming it "AlphabetSoupCharity_Optimisation.h5."

## Summary 

The goal was to achieve an accuracy level above 75% on a binary classification model. Despite multiple attempts, the maximum accuracy achieved was 73%. I couldn't reach the target accuracy of over 75%. The consistent results across different model variations suggest that other factors may be influencing the model's performance, necessitating further investigation and potentially additional data preprocessing or alternative modeling techniques. To increase model performance, I took several steps to optimize the structure, training process, and data handling for my neural network such as data cleaning, binning, increasing nodes, adding layers, changing activation functions, altering epochs, changing batch sizes, and more. 
To further improve accuracy of deep learning model, several strategies could be employed such:
* Experimenting more with the depth of the model - adding or removing more hidden layers to find the optimal depth of the model.
* Adjusting neurons per layer - experimenting with different number of neurons in each layer to balance complexity and performance.
* Test even more activation functions.
* Try different optimizers.

Given the current model's limitations, I recommend exploring different machine learning algorithms that might offer improved performance for this classification problem. One potential approach is to use *Random Forests* or *Gradient Boosting Machines (GBM)*, which are robust ensemble methods known for its effectivness in classification task. 
This recommendation comes from a wide research that I have conducted. Random Forest and GBM are ensemble techniques that combine multiple models to reduce overfitting and improve generalization. They can be more resistant to noise and decrease data loss in the current deep learning model. These methods provide insights into future learnings and can be valuable for refining the model and understanding the data better. Those methods are more robust to outliers and variations, which can overall improve accuracy and reliability of the model.