import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
)


def clip_raster_with_vector(
    raster_path, geojson_path, output_raster_path
):
    """
    Clips a raster file with a vector file and save the output to GeoTIFF.

    Arguments:
        raster_path (str): Input raster file path.
        geojson_path (str): GeoJSON path for clipping.
        output_raster_path (str): Output GeoTIFF path.

    Returns:
        None
    """
    # Read the vector file
    shapes = gpd.read_file(geojson_path)
    if shapes.geometry.empty:
        raise ValueError("The vector has no geometries for use in clipping the raster.")
    
    # Run Union
    geometries = [mapping(shapes.geometry.unary_union)]

    # Load the raster file and retrieve profile
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geometries, crop=True)
        out_meta = src.meta

    out_meta.update(
        {
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
            "dtype": rasterio.float64,
            "compress": "lzw"
        }
    )
    # Save
    with rasterio.open(output_raster_path, "w", **out_meta) as dest:
        dest.write(out_image.astype(rasterio.float64))


def convert_raster_to_array(raster_path):
    """
    Open a raster file and convert it to a numpy array.

    Arguments:
        raster_path (str): Raster file path.

    Returns:
        ndarray: Numpy array of raster file.
    """

    with rasterio.open(raster_path) as src:
        # Read raster file
        ras_array = src.read()
        ras_array = ras_array.astype(rasterio.float64)

        # Get No Data value
        nan_value = src.nodatavals[0]

        # Apply No Data Value
        if nan_value is None:
            return ras_array
        ras_array[ras_array == nan_value] = np.nan
        src.close()
        return ras_array

def split_data(df, class_column, strat_value, randomn):
    """
    Splits the data into training and validation sets with stratified sampling.

    Arguments:
        df (pd.DataFrame):  DataFrame to split into training and validation sets
        class_column (str): Name of column where the class value is stored
        strat_value (float): Proportion of the data that will be in the validation data
        randomn (int): Value that controls the shuffling applied to the data before the split

    Returns:
        tuple(pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame, 
        pd.DataFrame): a tuple containing the training and validation sets 

    """

    # Drop the class column
    x_df = df.drop(class_column, axis=1)
    y = df[class_column]

    # Split using sk-learn's train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x_df, y, 
                                                        test_size=strat_value,
                                                        random_state=randomn)
    
    return X_train, X_test, y_train, y_test

def scale_to_min_max(df_train, df_test):
    """
    Scale the input dataframes to 0-1.

    Arguments:
        df_train (pd.DataFrame): Training DataFrame
        df_test (pd.DataFrame): Test DataFrame

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): Scaled dataframes

    """
    # Instantiate the min-max scaler
    scaler = MinMaxScaler()

    # Scale the training dataframe
    df_train_scaled = scaler.fit_transform(df_train)
    df_train_scaled = pd.DataFrame(df_train_scaled, columns=df_train.columns)

    # Scale the test dataframe
    df_test_scaled = scaler.fit_transform(df_test)
    df_test_scaled = pd.DataFrame(df_test_scaled, columns=df_test.columns)
    
    return df_train_scaled, df_test_scaled
    

def run_pca(df_train, df_test, n_components):
    """
    Run Principal Components Analysis on DataFrames

    Arguments:
        df_train (pd.DataFrame): Training DataFrame
        df_test (pd.DataFrame): Test DataFrame
        n_components (int): Number of components

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): Transformed data using PCA

    """

    # Instantiate PCA
    pca = PCA(n_components=n_components)

    # Run PCA on training dataframe
    df_train_pca = pca.fit_transform(df_train)
    df_train_pca = pd.DataFrame(df_train_pca, 
                                columns=[f"PC_{component}" for component in range(n_components)]
                                )
    
    # Run PCA on test dataframe
    df_test_pca = pca.fit_transform(df_test)
    df_test_pca = pd.DataFrame(df_test_pca, 
                                columns=[f"PC_{component}" for component in range(n_components)]
                                )
    
    return df_train_pca, df_test_pca

def run_lda(df_train_x, df_train_y, df_test, n_components):
    """
    Run Linear Discriminant Analysis on DataFrames

    Arguments:
        df_train_x (pd.DataFrame): Training DataFrame
        df_train_y (pd.DataFrame): Training DataFrame containing the class
        df_test (pd.DataFrame): Test DataFrame
        n_components (int): Number of components

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): Transformed data using LDA

    """

    # Instantiate LDA
    lda = LinearDiscriminantAnalysis(n_components=n_components)

    # Run PCA on training dataframe
    df_train_lda = lda.fit_transform(df_train_x, df_train_y)
    df_train_lda = pd.DataFrame(df_train_lda, 
                                columns=[f"LDA_{component}" for component in range(n_components)]
                                )
    
    # Run PCA on test dataframe
    df_test_lda = lda.transform(df_test)
    df_test_lda = pd.DataFrame(df_test_lda, 
                                columns=[f"LDA_{component}" for component in range(n_components)]
                                )
    
    return df_train_lda, df_test_lda

def log_transform(df_train, df_test, columns):
    """
    Apply a log transformation on the input train and test dataframes.

    Arguments:
        df_train (pd.DataFrame): Training DataFrame
        df_test (pd.DataFrame): Test DataFrame
        columns (list): List of the numerical columns that will be transformed

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): Log transformed dataframes

    """

    # Instantiate the log transformer
    log_transformer = FunctionTransformer(func=np.log1p)

    # Transform the train dataset
    df_train_log = log_transformer.fit_transform(df_train[columns])
    df_train_log = pd.DataFrame(df_train_log, columns=columns, index=df_train.index)

    # Transform the test dataset
    df_test_log = log_transformer.transform(df_test[columns])
    df_test_log = pd.DataFrame(df_test_log, columns=columns, index=df_test.index)

    return df_train_log, df_test_log

def transform_df(df, poly, columns, fit=False):
    """
    Run a polynomial transform up to specified degree and columns

    Arguments:
        df_train (pd.DataFrame): DataFrame for transformation
        columns (list): List of the numerical columns that will be transformed
        poly (sk-learn): sk-learn PolynomialFeatures algorithm

    Returns:
        pd.DataFrame: Polynomial transformed dataframes
    """

    if df is None:
        return None
    
    # Subset to the features that will be transformed
    transform_features = df[columns]

    # Apply transformation
    transformed = (
        poly.fit_transform(transform_features)
        if fit
        else poly.transform(transform_features)
    )

    # Get names of features
    new_names = poly.get_feature_names_out(columns)
    transformed_df = pd.DataFrame(
        transformed, columns=new_names, index=df.index
    )

    # Get transformed features
    df_excluded = df.drop(columns=columns)

    return pd.concat([df_excluded, transformed_df], axis=1)

def poly_transform(df_train, df_test, columns, degree):
    """
    Apply a polynomial transformation on the input train and test dataframes.

    Arguments:
        df_train (pd.DataFrame): Training DataFrame
        df_test (pd.DataFrame): Test DataFrame
        columns (list): List of the numerical columns that will be transformed
        degree (int): the maximum degree of polynomial transformation

    Returns:
        tuple(pd.DataFrame, pd.DataFrame): Polynomial transformed dataframes

    """

    # Instantiate PolynomialFeatures
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    # Transform train dataframe
    df_train_transformed = transform_df(df_train, poly, columns, fit=True)
    
    # Transform test dataframe
    df_test_transformed = transform_df(df_test, poly, columns)

    return df_train_transformed, df_test_transformed