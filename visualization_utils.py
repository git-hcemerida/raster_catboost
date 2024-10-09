import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configure a default figure size
figure_size = (12, 8)

def make_pairplot(df, hue):
        """
        Create a pair plot to visualize the relationships between variables in a
        DataFrame.

        Arguments:
            df (pd.DataFrame): The DataFrame containing the variables to be
            plotted.
            hue (str): The column name in the DataFrame to be used for coloring
            the plot.

        Returns:
            pairplot: Plot showing the relationships between variables
        """
        
        # Configure plot size
        plt.figure(figsize=figure_size)

        # Generate the pair plot
        sns.pairplot(df, hue=hue)

        # Set plot title and display plot
        plt.title("Pair Plot")
        plt.tight_layout()
        plt.show()

def make_grouped_boxplot(df, hue):
        """
        Generate a grouped box plot of values.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            hue (str): The name of the column representing the hue variable.

        Returns:
            boxplot: Plot showing the box plot for each class and each band
        """
        
        # Configure plot size
        plt.figure(figsize=figure_size)

        # Melt the dataframe for compatibility
        newdf = pd.melt(df, id_vars=hue)

        # Generate the box plot separated by classes
        sns.boxplot(newdf, x='variable', y='value', hue=hue)

        # Set plot title and display plot
        plt.title("Grouped Box Plot")
        plt.tight_layout()
        plt.show()

def make_correlation_matrix(df):
        """
        Generate a heatmap through correlation matrix of the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame for which the heatmap needs to be
            generated.

        Returns:
            heatmap: Correlation matrix 
        """
        # Configure plot size
        plt.figure(figsize=figure_size)

        # Configure dataframe data type
        newdf = df.select_dtypes(include=[np.number])

        # Generate the correlation matrix and plot
        sns.heatmap(newdf.corr(), annot=True, cmap="coolwarm")

        # Set plot title and display plot
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()