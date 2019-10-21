"""
Author: Tanner Robart 
Date: August 2019 
Python version: Python 3.7
"""

import sys
import os
import warnings
import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import shap
import unittest
import logging

warnings.simplefilter("ignore")  # Suppress LightGBM warnings while doing unit tests


class BinaryLightGBMHandler:
    """Wrapper class to handle the functions of training and evaluating 
    a Binary LightGBM Classifier model given feature and target class data.
    """

    def __init__(
        self, seed=7, encode_categorical_method=False, impute_missing_values=False, params=None
    ):
        # Model attribute (self.__gbm) protected to prevent access except through handler class methods
        self.__gbm = lgb.LGBMClassifier(verbose=-1)
        self.seed = seed  # RNG seed for reproducibility
        self.imputation_mean = None
        self.encode_categorical_method = encode_categorical_method
        self.impute_missing_values = impute_missing_values
        self.category_encoder = None
        self.imputer = None
        self.categorical_columns = None
        self.numerical_columns = None
        if params is None:
            self.params = {
                "boosting_type": "gbdt",
                "max_depth": 10,
                "objective": "binary",
                "num_leaves": 40,
                "learning_rate": 0.05,
                "colsample_bytree": 0.8,
                "reg_alpha": 5,
                "reg_lambda": 10,
                "min_child_weight": 1,
                "min_child_samples": 5,
                "scale_pos_weight": 1,
                "metric": "logloss",
            }
        else:
            self.params = params

    def preprocess(self, X, y=None):
        """Handles common forms of missing values in X, replace with NaN.
        Also encodes the target y in case it is not already a numeric class.
        Returns processed X and y.

        LightGBM is supposed to have its own means to handle missing values, 
        so we optionally allow the NaNs (default) or impute a replacement.
        
        LightBGM also handles optimizing splits for categorical features 
        automatically in O(k * log(k)) time, so we feed it the raw 
        categories for now as a default. If wanted to use a different GBM, 
        or wanted to see if improvements to the model might be made by
        transforming the categoricals, we could one-hot encode these 
        categoricals for the model (at the risk of creating extremely sparse space which 
        some decisions trees do poorly on), or label_encode() them to an ordinal scale. 
        (at the risk of creating an artificial ordinal relationship for 
        categories where none exist).
        
        Future additions would be adding an option for datawig (https://github.com/awslabs/datawig) 
        deep learning imputation method for both categorical and numeric data, seems powerful.
        """
        # Drop any rows from X and y where there is no ground truth value in y to train or test on
        if y is not None:
            if y.isna().any():
                X = X.sort_index().loc[y.dropna().sort_index().index.values]
                y = y.dropna()

        # Separating out the numericals and categoricals for encoding and imputations methods
        numericals = X.select_dtypes(include=["int", "float64"])
        categoricals = X.select_dtypes(include=["object", "category"])
        categoricals = categoricals.astype("category")
        self.numerical_columns = list(numericals)

        # Optionally encode the categoricals
        if self.encode_categorical_method == "one_hot":  # Encode as one-hot
            # determines if it is training an encoder or applying the previously trained encoder at prediction time to test data
            if self.category_encoder is None:
                onehotencoder = OneHotEncoder()
                self.category_encoder = onehotencoder.fit(categoricals)
                categoricals = self.category_encoder.transform(categoricals)
            else:
                categoricals = self.category_encoder.transform(categoricals)

        elif self.encode_categorical_method == "ordinal":  # Encode as ordinal
            # determines if it is training an encoder or applying the previously trained encoder at prediction time to test data
            if self.category_encoder is None:
                label_encoder = LabelEncoder()
                self.category_encoder = label_encoder.fit(categoricals)
                categoricals = self.category_encoder.transform(categoricals)
            else:
                categoricals = self.category_encoder.transform(categoricals)

        # Optionally impute numericals to mean
        # Could add an option later to pass in different imputation values or methods (like datawig)
        if self.impute_missing_values == True:
            if self.imputer is None:
                imputer = SimpleImputer()
                self.imputer = imputer.fit(numericals)
                numericals = self.imputer.transform(numericals)
            else:
                numericals = self.imputer.transform(numericals)

        processed_x = pd.concat([categoricals, numericals], axis=1)

        if y is None:  # y wasn't passed as argument
            return processed_x
        else:
            # Making sure y is encoded as a binary class and not string
            label_encoder = LabelEncoder()
            encoded_y = label_encoder.fit_transform(y)
            return processed_x, encoded_y

    def fit(self, X, y):
        """Fits the model using X as features and y as ground truth. 
        Expects X, y are training data already split from full data
        """
        X, y = self.preprocess(X, y)
        self.__gbm.set_params(
            boosting_type="gbdt",
            objective="binary",
            n_jobs=3,
            silent=True,
            num_leaves=50,
            metric=self.params["metric"],
            max_depth=self.params["max_depth"],
            min_child_weight=self.params["min_child_weight"],
            min_child_samples=self.params["min_child_samples"],
            scale_pos_weight=self.params["scale_pos_weight"],
        )

        # Create list of categorical columns numbers
        categorical_features = [
            c
            for c, col in enumerate(X.columns)
            if col in list(X.select_dtypes(include=["category"]).columns)
        ]

        ## method utilizing the special lgb.Dataset and .train() method instead of .fit()
        # train_data = lgb.Dataset(X, label = y, categorical_feature = categorical_features)
        # self.__gbm = lgb.train(self.params,
        #                     train_data,
        #                     num_boost_round=10,
        #                     valid_sets=train_data,
        #                     categorical_feature = categorical_features)

        self.__gbm = self.__gbm.fit(X, y, categorical_feature=categorical_features)

    def feature_plots(self, X):
        """Provide some basic visualization of feature importance 
        on currently trained model, given a set of values X
        """
        # plots tree decision structure
        ax = lgb.plot_tree(self.__gbm)
        plt.rcParams["figure.figsize"] = [50, 10]
        plt.show()

        # Using shap for more detailed breakdown of feature importance values
        # (some ranges of a numerical feature are more important than others)
        shap.initjs()
        shap_values = shap.TreeExplainer(self.__gbm).shap_values(X)

        # More detailed breakdown of all SHAP values
        plt.figure()
        ax = shap.summary_plot(shap_values, X)
        plt.show()

        # The mean absolute value of the SHAP values for each feature to get a standard bar plot
        plt.figure()
        ax = shap.summary_plot(shap_values, X, plot_type="bar")
        plt.show()

    def predict(self, X):
        """Given a test set X of features, 
        produce an output array of predicted classifications
        """
        X = self.preprocess(X)  # Imputes and encodes based on training data
        return self.__gbm.predict(X)

    def predict_proba(self, X):
        """Given a test set X of features, produce an output
        array of the probabilities of each class per item in X
        """
        X = self.preprocess(X)  # Imputes and encodes based on training data
        return self.__gbm.predict_proba(X)

    def evaluate(self, X, y):
        """returns the f1 score and logloss performance metrics 
        of the current models predictions given a set of test_x and test_y
        """
        X, y = self.preprocess(X, y)
        y_true = y
        y_pred = self.predict(X)
        y_pred_prob = self.predict_proba(X)
        return {
            "f1_score": f1_score(y_true, y_pred, average="binary"),
            "logloss": log_loss(y_true, y_pred_prob),
        }

    def tune_parameters(self, X, y, params=None, k_folds=3):
        """Uses Kfold CV to determine the performance 
        of the GBM algo using a certain set of parameters. 
        """
        X, y = self.preprocess(X, y)

        if params is None:
            params = self.__gbm.get_params()
        else:
            self.__gbm.set_params(
                n_jobs=params["n_jobs"],
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                max_bin=params["max_bin"],
                max_depth=params["max_depth"],
                min_split_gain=params["min_split_gain"],
                min_child_weight=params["min_child_weight"],
                min_child_samples=params["min_child_samples"],
                scale_pos_weight=params["scale_pos_weight"],
            )

        cv_results = cross_validate(self.__gbm, X, y, cv=k_folds, scoring=["f1", "neg_log_loss"])

        params["scores"] = {
            "f1_score": np.mean(cv_results["test_f1"]),
            "logloss": np.mean(abs(cv_results["test_neg_log_loss"])),
        }

        return params

    def tune_hyperparameters(self, X, y, grid_params=None):
        """Performs a grid search to determine best parameters for model performance.
        future direction, would like to optionally switch from grid search to using the 
        hyperopt package (https://github.com/hyperopt/hyperopt) for more efficient 
        gradient descent of the hyperparameter space.
        """
        X, y = self.preprocess(X, y)
        default_grid_params = {
            "n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            "learning_rate": [0.01, 0.02, 0.05, 0.07, 0.10],
            "subsample": [0.3, 0.4, 0.5, 0.6, 0.7],
            "max_depth": [3, 4, 5, 6, 7, 8, 9],
            "colsample_bytree": [0.5, 0.45],
            "min_child_weight": [1, 2, 3],
            "alpha": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }

        if grid_params is None:
            grid_params = default_grid_params

        
        # Create the grid
        grid = GridSearchCV(self.__gbm, grid_params, verbose=1, cv=3, n_jobs=5)
        grid_results = grid.fit(X, y)  # Run the grid search

        # Return the best parameters found
        return grid_results.best_params_


class LGBMTestCase(unittest.TestCase):
    def setUp(self):
        """setUp() is a built-in function of unittest that re-initializes all of the given conditions, variables, and data before every test to prevent testing contamination
        """
        warnings.simplefilter("ignore")
        self.gbm_handler = BinaryLightGBMHandler()
        dirname = os.getcwd()
        filename = os.path.join(dirname, "DR_Demo_Lending_Club_reduced.csv")
        self.data = pd.read_csv(filename)
        X, y = self.data.iloc[:, 2:], self.data.iloc[:, 1]
        X.loc[X["emp_length"] == "na", "emp_length"] = np.nan  # setting 'na' to NaN
        # Hardcoding a typechange for 'emp_length' because pandas read_csv()
        # misclassifies this field as type == 'object' and otherwise hides 250 'na' values
        # that do not get picked up by pd.DataFrame's .isnull() method
        # do not want to hardcode this inside our class, so leaving it out here on an
        # assumption data handed to class will be properly typed when loaded.
        X = X.astype({"emp_length": "float64"})
        self.X = X
        self.y = y
        self.seed = 7
        self.test_size = 0.3

    def test_is_reproducible(self):

        logging.info("Testing reproducibility of model...\n")
        gbm_handler1 = self.gbm_handler
        gbm_handler2 = BinaryLightGBMHandler()

        # Splitting our data into test and train sets to be handled by our LightGBMHandler class
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=7
        )

        logging.info("Training....")
        gbm_handler1.fit(X_train, y_train)
        gbm_handler2.fit(X_train, y_train)
        logging.info("Evaluating....")
        model1_preds = gbm_handler1.predict(X_test)
        model2_preds = gbm_handler2.predict(X_test)
        model1_results = gbm_handler1.evaluate(X_test, y_test)
        model2_results = gbm_handler2.evaluate(X_test, y_test)

        self.assertTrue(np.array_equal(model1_preds, model2_preds))

        logging.info(
            "Model 1 Performance: ", model1_results, "\nModel 2 Performance: ", model2_results
        )
        self.assertEqual(model1_results, model2_results)
        logging.info("SUCCESS: Model is reproducible using same RNG seed!\n\n")

    def test_handles_missing_values(self):
        # alters X and y to contain missing NaN values

        logging.info("Testing missing value handling...\n")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=7, shuffle=False
        )
        # Randomly add nans
        X_train_change = X_train.sample(frac=0.3).index
        X_train.loc[X_train_change] = np.nan

        y_train_change = y_train.sample(frac=0.3).index
        y_train.loc[y_train_change] = np.nan

        X_test_change = X_test.sample(frac=0.3).index
        X_test.loc[X_test_change] = np.nan

        y_test_change = y_test.sample(frac=0.3).index
        y_test.loc[y_test_change] = np.nan

        logging.info("Training....\n")
        self.gbm_handler.fit(X_train, y_train)
        model_results = self.gbm_handler.evaluate(X_test, y_test)
        logging.info("SUCCESS: missing values handled!", "\n Model performance: ", model_results)

    def test_new_category_at_pred(self):
        logging.info("Testing new categories at prediction time handling... \n")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=7, shuffle=False
        )

        logging.info("Training.... \n")
        self.gbm_handler.fit(X_train, y_train)

        # Sprinkling in a new category level to the test data
        X_test_change = X_test.sample(frac=0.3).index
        X_test.loc[X_test_change, "home_ownership"] = "New category!"
        X_test.loc[X_test_change, "policy_code"] = "New category!"

        model_results = self.gbm_handler.evaluate(X_test, y_test)
        logging.info(
            "SUCCESS: new category level values handled!", "\n Model performance: ", model_results
        )

    def test_formatting(self):

        logging.info("Testing that methods return expected formats... \n")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=7, shuffle=False
        )

        self.gbm_handler.fit(X_train, y_train)
        predicts = self.gbm_handler.predict(X_test)
        predict_probas = self.gbm_handler.predict_proba(X_test)
        eval_results = self.gbm_handler.evaluate(X_test, y_test)
        tune_params_results = self.gbm_handler.tune_parameters(X_test, y_test)

        # Checks that predict() returns a numpy array of 1 dimension and is binary
        self.assertTrue(
            (type(predicts) == np.ndarray)
            & (predicts.ndim == 1)
            & ((predicts == 0) | (predicts == 1)).all(),
            msg="Format incorrect, output bad, bad programmer!",
        )

        logging.info("SUCCESS: BinaryLightGBMHandler() method predict() returns expected format!")

        # Checks that predict_proba() returns a numpy array of 2 dimension and contains only floats between 0 and 1
        self.assertTrue(
            (type(predict_probas) == np.ndarray)
            & (predict_probas.ndim == 2)
            & ((predict_probas > 0).all() & (predict_probas < 1).all()),
            msg="Format incorrect, output bad, bad programmer!",
        )

        logging.info(
            "SUCCESS: BinaryLightGBMHandler() method predict_proba() returns expected format!"
        )

        # Checks that evaluate() returns a dict of length 2 and is keyed as f1 and logloss
        self.assertTrue(
            (
                (type(eval_results) == dict)
                & (len(eval_results) == 2)
                & ("f1_score" in eval_results and "logloss" in eval_results)
            ),
            msg="Format incorrect, output bad, bad programmer!",
        )

        logging.info(
            "SUCCESS: BinaryLightGBMHandler() method evaluate() returns expected format!"
        )

        # Checks that tune_parameters() returns a dict of parameters and contains the average scores of the CV folds
        self.assertTrue(
            (
                (type(tune_params_results) == dict)
                & (len(tune_params_results["scores"]) == 2)
                & ("scores" in tune_params_results and "logloss" in tune_params_results["scores"])
            ),
            msg="Format incorrect, output bad, bad programmer!",
        )

        logging.info(
            "SUCCESS: BinaryLightGBMHandler() method tune_parameters() returns expected format!"
        )

    def test_parameter_tuning(self):
        # Alter test_params to overwrite any parameters of the model you want to modify for CV test
        logging.info("Testing tune_parameters() method... \n")
        test_params = {
            "n_jobs": 3,
            "num_boost_round": 100,
            "learning_rate": 0.01,
            "max_depth": -1,
            "num_leaves": 40,
            "max_bin": 128,
            "subsample_for_bin": 500,
            "min_split_gain": 0.5,
            "min_child_weight": 1,
            "min_child_samples": 5,
            "scale_pos_weight": 1,
        }

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, shuffle=True
        )
        self.gbm_handler.fit(X_train, y_train)
        cv_results = self.gbm_handler.tune_parameters(X_test, y_test, test_params, k_folds=3)
        logging.info(cv_results)

        self.assertTrue(type(cv_results) == dict)
        logging.info("SUCCESS: parameters scored using 3-fold cross validation")


    def test_hyperparameter_tuning(self):
        # Alter test_params to include any parameters you want to
        # optimize and what values to do optimization over
        # would like to add an option for hyperparameter optimization
        # using hyperopt library (https://github.com/hyperopt)
        test_params = {
            "n_estimators": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
            "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            "subsample": [0.3, 0.4, 0.5, 0.6, 0.7],
            "max_depth": [3, 4, 5, 6, 7, 8, 9],
            "colsample_bytree": [0.45, 0.50, 0.55],
            "min_child_weight": [1, 2, 3],
            "alpha": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, shuffle=True
        )
        self.gbm_handler.fit(X_train, y_train)
        grid_results = self.gbm_handler.tune_hyperparameters(self.X, self.y, test_params)
        logging.info(grid_results)

    def test_plot(self):
        """function runs a basic fit of the model on training data and then 
        makes some basic plots regarding model tree structure and feature importance
        """
        logging.info("Testing plot method...\n")
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=7
        )
        self.gbm_handler.fit(X_train, y_train)
        X_train, y_train = self.gbm_handler.preprocess(X_train, y_train)
        X_test, y_test = self.gbm_handler.preprocess(
            X_test, y_test
        )  # Preprocessing the test data so it matches the train dtypes

        self.gbm_handler.feature_plots(
            X_train
        )  # Plot various aspects of the model and its trained feature importances
        logging.info("SUCCESS: Plots made")


if __name__ == "__main__":
    unittest.main(__name__, argv=["main"], exit=False, verbosity=1)
