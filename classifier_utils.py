from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from catboost import CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm

def calculate_accuracy(
        y_true, y_pred
    ):
        """
        This function calculates classification metrics.

        Args:
            y_true: List: True labels
            y_pred: List: Predicted labels

        Returns:
            tuple: Accuracy, Precision, Recall, F1 Score
        """
        accuracy = accuracy_score(y_true, y_pred, normalize=True)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        return accuracy, precision, recall, f1

def select_final_features(df):
    """
    Subset a DataFrame to the selected features

    Arguments:
        df (np.array): DataFrame containing the preprocessed data

    Returns:
        np.array: DataFrame containing the final features

    """
    col_idx = [0, 4, 9, 14, 18, 51, 55, 57, 58, 60, 61, 64, 72, 77, 78]
    newdf = df[:, col_idx]
    return newdf

## Reference:
# https://github.com/catboost/catboost/blob/master/catboost/tutorials/feature_selection/select_features_tutorial.ipynb

def select_features(steps, train_pool, test_pool, nfeatures, 
                    algorithm = EFeaturesSelectionAlgorithm.RecursiveByShapValues):
    """
    Use CatBoost to determine the best features to retain when developing a model

    Arguments:
        algorithm: The algorithm to be used. Defaults to EFeaturesSelectionAlgorithm
        steps (int): The number of steps 
        train_pool (Pool): Training Pool
        test_pool (Pool): Test Pool 
        nfeatures (int): The number of features that will be selected

    Retuns:
        list: Names of the selected features
    """

    print('Algorithm:', algorithm)

    # Instantiate CatBoost
    model = CatBoostClassifier(iterations=500, random_seed=0)

    # Run the algorithm to select features
    summary = model.select_features(
        train_pool,
        eval_set=test_pool,
        features_for_select=list(range(train_pool.num_col())),
        num_features_to_select=nfeatures,
        steps=steps,
        algorithm=algorithm,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=True,
        logging_level='Silent',
        plot=True
    )
    print('Selected features:', summary['selected_features_names'])
    return summary