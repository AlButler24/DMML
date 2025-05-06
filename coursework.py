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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.manifold import LocallyLinearEmbedding

        



# Please write the optimal hyperparameter values you obtain in the global variable 'optimal_hyperparm' below. This
# variable should contain the values when I look at your submission. I should not have to run your code to populate this
# variable.
optimal_hyperparam = {
    'hidden_layer_sizes': (50,),
    'activation': 'relu',
    'alpha': 1.0,
    'batch_size': 256,
    'learning_rate': 'adaptive'
}

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
 
        scaler = StandardScaler()

        original_shape = inp.shape
        if len(original_shape) > 2:
            inp_2d = inp.reshape(original_shape[0], -1)
        else:
            inp_2d = inp
        
        standardized_data = scaler.fit_transform(inp_2d)
        
        standardized_data *= 2.5
        
        if len(original_shape) > 2:
            standardized_data = standardized_data.reshape(original_shape)
        
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

        if test_size is None:
            test_size = 0.3

        if pre_split_data is not None:
            x_train, x_test, y_train, y_test = pre_split_data
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                self.x, self.y, test_size=test_size, random_state=42, stratify=self.y
            )

        x_train_std, scaler = self.q2(x_train)
        x_test_std = scaler.transform(x_test) * 2.5

        if hyperparam is None:
            hyperparam = {
                'hidden_layer_sizes': [(50,), (100,), (50, 25)],
                'activation': ['relu', 'tanh'],
                'learning_rate': ['adaptive'],
                'batch_size': [64]
            }

        default_params = {
            'alpha': 0.1,  
            'max_iter': 10, 
            'random_state': 42,
            'solver': 'adam',
            'early_stopping': False,
            'validation_fraction': 0.2
        }

        current_best_params = {
            key: values[0] for key, values in hyperparam.items()
        }
        current_best_params.update(default_params)

        self.param_performance = {}

        for param in hyperparam:
            best_local_value = None
            local_best_score = -np.inf
            self.param_performance[param] = {}

            for value in hyperparam[param]:
                trial_params = current_best_params.copy()
                trial_params[param] = value

                model = MLPClassifier(**trial_params)
                model.fit(x_train_std, y_train)
                score = model.score(x_test_std, y_test)

                self.param_performance[param][value] = score

                if score > local_best_score:
                    local_best_score = score
                    best_local_value = value

            current_best_params[param] = best_local_value

        global optimal_hyperparam
        optimal_hyperparam = current_best_params.copy()

        res1 = MLPClassifier(**current_best_params)
        res1.max_iter = 1
        res1.warm_start = True

        max_iter = 500
        res2 = np.zeros(max_iter)
        res3 = np.zeros(max_iter)
        res4 = np.zeros(max_iter)

        for i in range(max_iter):
            res1.fit(x_train_std, y_train)

            if hasattr(res1, 'loss_curve_') and len(res1.loss_curve_) > 0:
                res2[i] = res1.loss_curve_[-1]
            else:
                res2[i] = np.nan

            res3[i] = res1.score(x_train_std, y_train)
            res4[i] = res1.score(x_test_std, y_test)

            if i >= 10:
                recent_test_acc = res4[max(0, i - 10):i + 1]
                if np.std(recent_test_acc) < 0.001:
                    res2 = res2[:i + 1]
                    res3 = res3[:i + 1]
                    res4 = res4[:i + 1]
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
        
        alpha_values = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
        
        try:
            best_params = optimal_hyperparam.copy()
        except NameError:
            best_params = {
                'hidden_layer_sizes': (50,),
                'activation': 'relu',
                'solver': 'adam',
                'batch_size': 64,
                'learning_rate': 'adaptive',
                'max_iter': 200,
                'random_state': 42
            }
        
        x_train, x_test, y_train, y_test = train_test_split(
            self.x, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        x_train_std, scaler = self.q2(x_train)
        x_test_std = scaler.transform(x_test) * 2.5
        
        n_alphas = len(alpha_values)
        train_scores = np.zeros(n_alphas)
        test_scores = np.zeros(n_alphas)
        avg_param_magnitudes = np.zeros(n_alphas)
        losses = np.zeros(n_alphas)
        iterations = np.zeros(n_alphas)
        
        for i, alpha in enumerate(alpha_values):
            current_params = best_params.copy()
            current_params['alpha'] = alpha
            
            model = MLPClassifier(**current_params)
            model.fit(x_train_std, y_train)
            
            train_scores[i] = model.score(x_train_std, y_train)
            test_scores[i] = model.score(x_test_std, y_test)
            
            if hasattr(model, 'loss_curve_') and len(model.loss_curve_) > 0:
                losses[i] = model.loss_curve_[-1]
            
            iterations[i] = model.n_iter_

            param_magnitudes = []
            for layer in range(len(model.coefs_)):
                param_magnitudes.append(np.abs(model.coefs_[layer]).mean())
                param_magnitudes.append(np.abs(model.intercepts_[layer]).mean())
            
            avg_param_magnitudes[i] = np.mean(param_magnitudes)
        
        generalization_gap = train_scores - test_scores
        
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
        global optimal_hyperparam

        try:
            best_params = optimal_hyperparam.copy()
        except (NameError, AttributeError):
            best_params = {
                'hidden_layer_sizes': (50,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.1,
                'batch_size': 64,
                'learning_rate': 'adaptive',
                'max_iter': 200,
                'random_state': 42
            }
        
        mlp = MLPClassifier(**best_params)
        
        std_x, _ = self.q2(self.x)
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        regular_cv_scores = cross_val_score(mlp, std_x, self.y, cv=kf)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        stratified_cv_scores = cross_val_score(mlp, std_x, self.y, cv=skf)
        
        stratified_cv_accuracy = np.mean(stratified_cv_scores)
        regular_cv_accuracy = np.mean(regular_cv_scores)
        
        n = len(stratified_cv_scores)
        diff = stratified_cv_scores - regular_cv_scores
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        t_stat = mean_diff / (std_diff / np.sqrt(n))
        
        df = n - 1
        
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        if p_value < 0.05:
            result = 'Splitting method impacted performance'
        else:
            result = 'Splitting method had no effect'
        
        res1 = stratified_cv_accuracy
        res2 = regular_cv_accuracy
        res3 = p_value
        res4 = result
        
        return res1, res2, res3, res4


    def q6(self):
        """
        This function should perform unsupervised learning using LocallyLinearEmbedding in Sklearn. You can assume that
        the function 'q1' has been called prior to calling this function.

        :return: The function should return the data you visualize.
        """
        # Standardize input data
        x_std, _ = self.q2(self.x)

        # Apply LLE
        lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
        x_lle = lle.fit_transform(x_std)

        # Store result in required variable name
        res = {
            'embedding': x_lle,
            'labels': self.y
        }

        return res