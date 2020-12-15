Building Artificial Neural Network  with TensorFlow-2.0

Data set customer churn modeling.csv in which a bank has given a fictional data based on these features we need to identify whether the customer is going to stay with the bank or to close the account and leaves the bank

Customer_Churn_Modelling - A model to check wheather the customer ceases his or her relationship with a company or not. 

*** Following steps are taken for building the project
        1. Importing require libraries(tensorflow,numpy,pandas,sklearn)
        
        2. Data Processing:
        Reading the csv file into pandas dataframe
        Excluding the irrelevant columns from the data
        separating target column from the data to a target variable 'y'
        
        As Artificial neural network works on a numerical data not on a string data so we have to use map function 
        to map this data into a numerical data (label encoder or one hot encoder)
       
        3. So using labelencoder in geography and gender column
        as, we can see categorical values in geography column so, using one hot encoding method to convert these values into continuous values 
        with the help of pandas get dummies function.
        
        4. Splitting the input(X) and feature(y) data into X_train, X_test and y_train, y_test and standardizing Feature
        (Standardize features by removing the mean and scaling to unit variance) to scale the values between -1 to 1
        
        
        5. Building Artificial neural network(sequencial network) --> Adding input layer, Hidden Layers with activation function
        6. Selecting adam Optimizer, binary_crossentropy Loss, and Performance Metrics and Compiling the model
        7. using model.fit to train the model
        8. getting 86.29% training accuracy 
        9. Predicting and Evaluating the model with y_test, y_pred and getting 0.8605 or 86% accuracy 
        
        Note: Adjust optimization parameters or model to improve accuracy
