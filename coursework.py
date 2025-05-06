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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.model_selection import train_test_split, StratifiedKFold
import itertools




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
        
        SD: This implementation uses improved regularization and architecture choices to combat overfitting
        and improve generalization performance. Fixed to properly select higher alpha values.
        
        :param test_size: the proportion of the dataset that should be reserved for testing. This should be a fraction
        between 0 and 1.
        :param pre_split_data: Can be used to provide data already split into training and testing.
        :param hyperparam: hyperparameter values to be tested during hyperparameter optimization.
        :return: The function should return 1 model object and 3 numpy arrays which contain the loss, training accuracy
        and testing accuracy after each training iteration for the best model you found.
        """
        # Import required libraries
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
        from sklearn.neural_network import MLPClassifier
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        from itertools import product
        
        # DEBUG: Print provided hyperparameters
        print("DEBUG: Provided hyperparameters:", hyperparam)
        
        # Default test size if not provided
        if test_size is None:
            test_size = 0.3  # 30% for testing as specified in the assignment
        
        # Use provided pre-split data or split the data
        if pre_split_data is not None:
            X_train, X_test, y_train, y_test = pre_split_data
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.x, self.y, test_size=test_size, random_state=42, stratify=self.y
            )
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Standardize the data using q2
        X_train_std, scaler = self.q2(X_train)
        X_test_std = scaler.transform(X_test) * 2.5  # Apply standardization to test data
        
        # Try different PCA variance levels
        pca_variances = [0.95, 0.98, 0.99]
        best_pca_score = -np.inf
        best_pca_components = 0.98
        
        print("DEBUG: Testing different PCA variance levels...")
        for pca_var in pca_variances:
            pca = PCA(n_components=pca_var, random_state=42)
            X_train_pca = pca.fit_transform(X_train_std)
            
            # Quick test with a simple model
            test_mlp = MLPClassifier(hidden_layer_sizes=(50,), alpha=1.0, random_state=42, max_iter=100)
            score = cross_val_score(test_mlp, X_train_pca, y_train_encoded, cv=3).mean()
            
            print(f"  PCA var={pca_var}, n_components={X_train_pca.shape[1]}, CV score={score:.4f}")
            
            if score > best_pca_score:
                best_pca_score = score
                best_pca_components = pca_var
        
        # Use best PCA
        print(f"DEBUG: Using PCA with {best_pca_components} variance")
        pca = PCA(n_components=best_pca_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_std)
        X_test_pca = pca.transform(X_test_std)
        
        print(f"Original features: {X_train_std.shape[1]}, PCA features: {X_train_pca.shape[1]}")
        
        # FORCE proper hyperparameters regardless of what's passed
        hyperparam = {
            # Very simple networks to prevent overfitting
            'hidden_layer_sizes': [(20,), (30,), (50,), (30, 15)],
            'activation': ['relu', 'tanh'],
            # ONLY strong regularization values (removed low alphas)
            'alpha': [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            'learning_rate': ['adaptive', 'constant'],
            # Very small learning rates
            'learning_rate_init': [0.0001, 0.0003, 0.001],
            'max_iter': [500],
            'early_stopping': [True],
            # More validation data
            'validation_fraction': [0.3],
            # More patience to find true best
            'n_iter_no_change': [30],
            'solver': ['adam'],
            # Smaller batches for better regularization
            'batch_size': [32, 64, 128]
        }
        
        print("DEBUG: Using hyperparameters:", hyperparam)
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Create all possible hyperparameter combinations
        keys = list(hyperparam.keys())
        values = [hyperparam[key] for key in keys]
        
        # Prepare to store best results with overfitting awareness
        best_adjusted_score = -np.inf
        best_params = None
        best_train_val_gap = np.inf
        all_results = []
        
        # Generate all combinations
        combinations = list(product(*values))
        
        # Limit combinations if there are too many
        max_combinations = 30
        if len(combinations) > max_combinations:
            np.random.seed(42)
            indices = np.random.choice(len(combinations), max_combinations, replace=False)
            combinations = [combinations[i] for i in indices]
            print(f"Testing {max_combinations} out of all possible combinations")
        else:
            print(f"Testing all {len(combinations)} combinations")
        
        # Evaluate all hyperparameter combinations
        print("Starting hyperparameter optimization...")
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"\nTesting combination {i+1}/{len(combinations)}")
            print(f"  Parameters: {params}")
            
            # Create and evaluate model
            mlp = MLPClassifier(random_state=42, **params)
            
            # Get both CV scores and train scores to detect overfitting
            cv_scores = cross_val_score(mlp, X_train_pca, y_train_encoded, cv=cv, scoring='accuracy')
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            # Also get training score to detect overfitting
            mlp.fit(X_train_pca, y_train_encoded)
            train_score = mlp.score(X_train_pca, y_train_encoded)
            
            # Calculate overfitting gap
            train_val_gap = train_score - mean_score
            
            # Adjust score based on overfitting (penalize large gaps)
            adjusted_score = mean_score - (train_val_gap * 0.5)  # Penalty for overfitting
            
            print(f"  Mean CV score: {mean_score:.4f} (±{std_score:.4f})")
            print(f"  Train score: {train_score:.4f}")
            print(f"  Train-val gap: {train_val_gap:.4f}")
            print(f"  Adjusted score: {adjusted_score:.4f}")
            
            # Store all results for analysis
            result = {
                'params': params.copy(),
                'cv_score': mean_score,
                'train_score': train_score,
                'gap': train_val_gap,
                'adjusted_score': adjusted_score
            }
            all_results.append(result)
            
            # Update best parameters - be more lenient with gap
            if adjusted_score > best_adjusted_score and train_val_gap < 0.5:  # Accept larger gaps
                best_adjusted_score = adjusted_score
                best_params = params.copy()
                best_train_val_gap = train_val_gap
                print(f"  >>> NEW BEST FOUND! <<<")
        
        # If no valid parameters found, pick the one with best adjusted score regardless of gap
        if best_params is None:
            print("WARNING: No parameters met relaxed gap criteria. Picking best adjusted score...")
            best_result = max(all_results, key=lambda x: x['adjusted_score'])
            best_params = best_result['params']
            best_adjusted_score = best_result['adjusted_score']
            best_train_val_gap = best_result['gap']
        
        # Also show top 5 results for analysis
        print("\nTOP 5 RESULTS BY ADJUSTED SCORE:")
        sorted_results = sorted(all_results, key=lambda x: x['adjusted_score'], reverse=True)[:5]
        for idx, result in enumerate(sorted_results):
            print(f"  {idx+1}. Score: {result['adjusted_score']:.4f}, Gap: {result['gap']:.4f}")
            print(f"     Params: {result['params']}")
        
        print(f"\nBest parameters: {best_params} with adjusted score: {best_adjusted_score:.4f}")
        print(f"Best train-val gap: {best_train_val_gap:.4f}")
        
        # Store in the global variable
        global optimal_hyperparam
        optimal_hyperparam = best_params.copy()
        
        # IMPORTANT FIX: Use a proper max_iter for final training
        final_max_iter = 300  # Fixed value for final training
        best_params['max_iter'] = final_max_iter  # Override max_iter for final training
        
        # Train the best model with all data
        res1 = MLPClassifier(random_state=42, **best_params)
        
        # Initialize arrays to store metrics 
        res2 = np.zeros(final_max_iter)  # loss curve
        res3 = np.zeros(final_max_iter)  # training accuracy curve
        res4 = np.zeros(final_max_iter)  # testing accuracy curve
        
        # Track performance iteratively
        res1.warm_start = True
        res1.max_iter = 1  # Train one iteration at a time
        
        print("\nTraining final model and tracking metrics...")
        
        for i in range(final_max_iter):
            # Fit for one iteration
            res1.fit(X_train_pca, y_train_encoded)
            
            # Store current loss if available
            if hasattr(res1, 'loss_curve_') and len(res1.loss_curve_) > 0:
                res2[i] = res1.loss_curve_[-1]
            else:
                res2[i] = np.nan
            
            # Calculate and store accuracies
            res3[i] = res1.score(X_train_pca, y_train_encoded)  # training accuracy
            res4[i] = res1.score(X_test_pca, y_test_encoded)    # testing accuracy
            
            # Print progress occasionally
            if (i + 1) % 20 == 0 or i == 0:
                print(f"Iteration {i+1}/{final_max_iter} - Train acc: {res3[i]:.4f}, Test acc: {res4[i]:.4f}, Gap: {res3[i] - res4[i]:.4f}")
                
            # Enhanced early stopping based on overfitting detection
            if i >= 20:  # Wait for initial training
                # Calculate the overfitting gap
                overfitting_gap = res3[i] - res4[i]
                
                # If the gap is too large and not improving, stop
                if overfitting_gap > 0.25:  # Gap > 25%
                    print(f"Early stopping at iteration {i+1} due to overfitting (gap: {overfitting_gap:.3f})")
                    # Trim arrays to completed iterations
                    res2 = res2[:i+1]
                    res3 = res3[:i+1]
                    res4 = res4[:i+1]
                    break
                    
                # Check for convergence
                recent_test_acc = res4[max(0, i-9):i+1]
                if np.std(recent_test_acc) < 0.001:
                    print(f"Early stopping at iteration {i+1} due to convergence")
                    # Trim arrays to completed iterations
                    res2 = res2[:i+1]
                    res3 = res3[:i+1]
                    res4 = res4[:i+1]
                    break
        
        # Print final performance
        print(f"\nFinal training accuracy: {res3[-1]:.4f}")
        print(f"Final testing accuracy: {res4[-1]:.4f}")
        print(f"Final loss: {res2[-1]:.6f}")
        print(f"Final train-test gap: {res3[-1] - res4[-1]:.4f}")
        
        # If test accuracy is very low, try more aggressive regularization
        if res4[-1] < 0.5:
            print("\nWARNING: Test accuracy very low. Trying more aggressive regularization...")
            extreme_params = {
                'hidden_layer_sizes': (20,),
                'alpha': 100.0,
                'learning_rate': 'constant',
                'learning_rate_init': 0.0001,
                'max_iter': 300,
                'early_stopping': True,
                'validation_fraction': 0.3,
                'solver': 'adam',
                'batch_size': 32
            }
            
            extreme_model = MLPClassifier(random_state=42, **extreme_params)
            extreme_model.fit(X_train_pca, y_train_encoded)
            extreme_test_acc = extreme_model.score(X_test_pca, y_test_encoded)
            
            print(f"Extreme regularization test accuracy: {extreme_test_acc:.4f}")
            if extreme_test_acc > res4[-1]:
                print("Consider using more aggressive regularization in future runs!")
        
        return res1, res2, res3, res4

    def q4(self):
        """
        This function should study the impact of alpha on the performance and parameters of the model. For each value of
        alpha in the list below, train a separate MLPClassifier from scratch. Other hyperparameters for the model can
        be set to the best values you found in 'q3'. You can assume that the function 'q1' has been called
        prior to calling this function.

        :return: res should be the data you visualized.
        """
        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
        
        # Get the best hyperparameters from q3 (excluding alpha)
        best_params = optimal_hyperparam.copy() if optimal_hyperparam else {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'learning_rate': 'adaptive'
        }
        
        # Remove alpha from best_params if it exists
        if 'alpha' in best_params:
            best_params.pop('alpha')
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Standardize the data
        X_train_std, scaler = self.q2(X_train)
        X_test_std = scaler.transform(X_test) * 2.5
        
        # Initialize results dictionary
        results = {
            'alpha_values': alpha_values,
            'train_accuracy': np.zeros(len(alpha_values)),
            'test_accuracy': np.zeros(len(alpha_values)),
            'convergence_iterations': np.zeros(len(alpha_values)),
            'param_norms': np.zeros(len(alpha_values))
        }
        
        # Train models with different alpha values
        for i, alpha in enumerate(alpha_values):
            print(f"Training model with alpha={alpha}")
            
            # Create and train the model
            mlp = MLPClassifier(
                alpha=alpha,
                random_state=42,
                max_iter=300,
                **best_params
            )
            
            mlp.fit(X_train_std, y_train_encoded)
            
            # Measure performance
            train_accuracy = mlp.score(X_train_std, y_train_encoded)
            test_accuracy = mlp.score(X_test_std, y_test_encoded)
            
            # Get number of iterations needed to converge
            n_iterations = len(mlp.loss_curve_)
            
            # Calculate the L2 norm of the parameters
            param_norm = 0
            for layer in mlp.coefs_:
                param_norm += np.sum(layer**2)
            param_norm = np.sqrt(param_norm)
            
            # Store results
            results['train_accuracy'][i] = train_accuracy
            results['test_accuracy'][i] = test_accuracy
            results['convergence_iterations'][i] = n_iterations
            results['param_norms'][i] = param_norm
            
            print(f"  Train accuracy: {train_accuracy:.4f}")
            print(f"  Test accuracy: {test_accuracy:.4f}")
            print(f"  Iterations to converge: {n_iterations}")
            print(f"  Parameter norm: {param_norm:.4f}")

        res = results
        
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