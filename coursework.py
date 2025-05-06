import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
import itertools
from sklearn.model_selection import train_test_split, cross_validate
from itertools import product



# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.
optimal_hyperparam = {}

class COC131:
    def q1(self, filename=None):
        """
        This function should be used to load the data. To speed-up processing in later steps, lower resolution of the
        image to 32*32. The folder names in the root directory of the dataset are the class names. After loading the
        dataset, you should save it into an instance variable self.x (for samples) and self.y (for labels). Both self.x
        and self.y should be numpy arrays of dtype float.

        :param filename: this is the name of an actual random image in the dataset. You don't need this to load the
        dataset. This is used by me for testing your implementation.
        :return res1: a one-dimensional numpy array containing the flattened low-resolution image in file 'filename'.
        Flatten the image in the row major order. The dtype for the array should be float.
        :return res2: a string containing the class name for the image in file 'filename'. This string should be same as
        one of the folder names in the originally shared dataset.
        """

        dataset_path = "../dataset"
        images = []
        labels = []
        
        specific_img = None
        specific_label = ""
    
        class_names = [folder for folder in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, folder)) and not folder.startswith('.')]

        for class_name in class_names:
            class_path = os.path.join(dataset_path, class_name)
            
            for img_file in os.listdir(class_path):
                # Skip hidden files
                if img_file.startswith('.'):
                    continue
                    
                img_path = os.path.join(class_path, img_file)
                
                try:
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    img = img.resize((32, 32), Image.LANCZOS)
                    
                    img_array = np.array(img, dtype=float).reshape(-1)

                    images.append(img_array)
                    labels.append(class_name)
                    
                    if filename and os.path.basename(img_path) == filename:
                        specific_img = img_array
                        specific_label = class_name
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        self.x = np.array(images, dtype=float)
        self.y = np.array(labels)
        
        if filename:
            res1 = specific_img
            res2 = specific_label
        else:
            res1 = np.zeros(1)
            res2 = ''
        
        return res1, res2


    def q2(self, inp):
        """
        This function should compute the standardized data from a given 'inp' data. The function should work for a
        dataset with any number of features.

        :param inp: an array from which the standardized data is to be computed.
        :return res2: a numpy array containing the standardized data with standard deviation of 2.5. The array should
        have the same dimensions as the original data
        :return res1: sklearn object used for standardization.
        """
        # Create a StandardScaler object
        scaler = StandardScaler()
        
        # Fit the scaler to the input data
        # For multidimensional data, ensure it's 2D for StandardScaler
        original_shape = inp.shape
        if len(original_shape) > 2:
            inp_2d = inp.reshape(original_shape[0], -1)
        else:
            inp_2d = inp
        
        # Fit and transform the data
        standardized_data = scaler.fit_transform(inp_2d)
        
        # Scale to standard deviation of 2.5 (default is 1.0)
        standardized_data *= 2.5
        
        # Reshape back to original dimensions if needed
        if len(original_shape) > 2:
            standardized_data = standardized_data.reshape(original_shape)
        
        # Assign to the required return variables
        res1 = scaler
        res2 = standardized_data
        
        return res2, res1

    def q3(self, test_size=None, pre_split_data=None, hyperparam=None):
        """
        This function should build a MLP Classifier using the dataset loaded in function 'q1' and evaluate model
        performance. You can assume that the function 'q1' has been called prior to calling this function. This function
        should support hyperparameter optimizations.

        SD: This implementation finds the optimal MLP classifier with hyperparameter optimization 
        without using GridSearchCV, focusing on model generalization and performance tracking.

        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found.
        """
        # Import required libraries
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
        import numpy as np
        import itertools

        # Default test size if not provided
        if test_size is None:
            test_size = 0.3  # 30% for testing as specified in the assignment
        
        # Use provided pre-split data or split the data
        if pre_split_data is not None:
            x_train, x_test, y_train, y_test = pre_split_data
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                self.x, self.y, test_size=test_size, random_state=42, stratify=self.y
            )
        
        # Standardize the data using q2
        x_train_std, scaler = self.q2(x_train)
        x_test_std = scaler.transform(x_test) * 2.5  # Apply standardization to test data

        # Define hyperparameters to test if not provided
        if hyperparam is None:
            hyperparam = {
                'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                'activation': ['relu', 'tanh'],
                'alpha': [1.0],  # Fixed alpha as required
                'learning_rate': ['adaptive'],
                'batch_size': [64]
            }
        
        # Default values for non-optimized parameters
        default_params = {
            'max_iter': 200,
            'random_state': 42,
            'solver': 'adam',
            'early_stopping': False,  
            'validation_fraction': 0.2
        }
        
        # Get all hyperparameter combinations using itertools
        param_names = list(hyperparam.keys())
        param_values = list(hyperparam.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Initialize arrays to track performance
        best_score = -np.inf
        best_params = None
        
        # Manually evaluate each combination - avoids cross_validate issues
        for combo in param_combinations:
            # Create parameter dictionary for current combination
            current_params = dict(zip(param_names, combo))
            current_params.update(default_params)
            
            # Create and evaluate model
            model = MLPClassifier(**current_params)
            model.fit(x_train_std, y_train)
            
            # Get validation score to evaluate performance
            score = model.score(x_test_std, y_test)
            
            # Update best parameters if current is better
            if score > best_score:
                best_score = score
                best_params = current_params.copy()
        
        # Store optimal hyperparameters
        global optimal_hyperparam
        optimal_hyperparam = best_params.copy()
        
        # Create final model with best parameters
        res1 = MLPClassifier(**best_params)
        
        # Prepare for incremental training and tracking
        max_iter = 500
        res1.max_iter = 1
        res1.warm_start = True
        
        # Initialize arrays for tracking
        res2 = np.zeros(max_iter)  # loss curve
        res3 = np.zeros(max_iter)  # training accuracy
        res4 = np.zeros(max_iter)  # testing accuracy
        
        # Train incrementally and track metrics
        for i in range(max_iter):
            res1.fit(x_train_std, y_train)
            
            # Store loss
            if hasattr(res1, 'loss_curve_') and len(res1.loss_curve_) > 0:
                res2[i] = res1.loss_curve_[-1]
            else:
                res2[i] = np.nan
            
            # Store accuracies
            res3[i] = res1.score(x_train_std, y_train)
            res4[i] = res1.score(x_test_std, y_test)
            
            # Simple early stopping based on convergence
            if i >= 10:
                recent_test_acc = res4[max(0, i-10):i+1]
                if np.std(recent_test_acc) < 0.001:
                    # Trim arrays to completed iterations
                    res2 = res2[:i+1]
                    res3 = res3[:i+1]
                    res4 = res4[:i+1]
                    break
        
        return res1, res2, res3, res4

    def q4(self):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.
        
        SD: This function analyses how different alpha values affect model performance and parameters
        while keeping other hyperparameters constant. It returns a structured data object containing
        multiple metrics for visualization in the notebook.

        :return: res should be the data you visualized.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
        import numpy as np
        
        # List of alpha values to test
        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
        
        # Get the best hyperparameters from q3
        # Access from global variable if it exists, otherwise use defaults
        try:
            best_params = optimal_hyperparam.copy()
        except NameError:
            # Default hyperparameters if q3 hasn't been run
            best_params = {
                'hidden_layer_sizes': (50,),
                'activation': 'relu',
                'solver': 'adam',
                'batch_size': 64,
                'learning_rate': 'adaptive',
                'max_iter': 200,
                'random_state': 42
            }
        
        # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Standardize data using q2
        x_train_std, scaler = self.q2(x_train)
        x_test_std = scaler.transform(x_test) * 2.5
        
        # Arrays to store results
        n_alphas = len(alpha_values)
        train_scores = np.zeros(n_alphas)
        test_scores = np.zeros(n_alphas)
        avg_param_magnitudes = np.zeros(n_alphas)
        losses = np.zeros(n_alphas)
        iterations = np.zeros(n_alphas)
        
        # Train a model for each alpha value
        for i, alpha in enumerate(alpha_values):
            # Update parameters with current alpha value
            current_params = best_params.copy()
            current_params['alpha'] = alpha
            
            # Train model
            model = MLPClassifier(**current_params)
            model.fit(x_train_std, y_train)
            
            # Record performance metrics
            train_scores[i] = model.score(x_train_std, y_train)
            test_scores[i] = model.score(x_test_std, y_test)
            
            # Record final loss
            if hasattr(model, 'loss_curve_') and len(model.loss_curve_) > 0:
                losses[i] = model.loss_curve_[-1]
            
            # Record number of iterations to convergence
            iterations[i] = model.n_iter_
            
            # Calculate average magnitude of model parameters (weights and biases)
            param_magnitudes = []
            for layer in range(len(model.coefs_)):
                # Add average magnitude of weights for this layer
                param_magnitudes.append(np.abs(model.coefs_[layer]).mean())
                # Add average magnitude of biases for this layer
                param_magnitudes.append(np.abs(model.intercepts_[layer]).mean())
            
            # Store the average parameter magnitude across all layers
            avg_param_magnitudes[i] = np.mean(param_magnitudes)
        
        # Calculate generalization gap (training accuracy - testing accuracy)
        generalization_gap = train_scores - test_scores
        
        # Compile all results into a structured data object
        res = {
            'alpha_values': np.array(alpha_values),
            'train_scores': train_scores,
            'test_scores': test_scores,
            'param_magnitudes': avg_param_magnitudes,
            'losses': losses,
            'iterations': iterations,
            'generalization_gap': generalization_gap
        }
        
        return res

    def q5(self):
        """
        This function should perform hypothesis testing to study the impact of using CV with and without Stratification
        on the performance of MLPClassifier. Set other model hyperparameters to the best values obtained in the previous
        questions. Use 5-fold cross validation for this question. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: The function should return 4 items - the final testing accuracy for both methods of CV, p-value of the
        test and a string representing the result of hypothesis testing. The string can have only two possible values -
        'Splitting method impacted performance' and 'Splitting method had no effect'.
        """

        res1 = 0
        res2 = 0
        res3 = 0
        res4 = ''

        return res1, res2, res3, res4

    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.
        """

        res = np.zeros(1)

        return res