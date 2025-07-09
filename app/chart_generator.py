# chart_generator.py
# 📊 Core chart rendering logic using matplotlib

import os
import pandas as pd
import matplotlib.pyplot as plt
from app.core.config import FILES_ROOT, CHART_TYPES

# === 📦 Utility Functions === #

# --- Function: load_table_data ---
def load_table_data(dataset_name, table_name):
    """
    Load a CSV file for a given dataset/table into a pandas DataFrame.
    Raises FileNotFoundError if the file does not exist.
    """
    path = os.path.join(FILES_ROOT, dataset_name, f"{table_name}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        raise FileNotFoundError(f"CSV for table '{table_name}' not found at {path}")


# --- Function: list_tables_from_dataset ---
def list_tables_from_dataset(dataset_name):
    """
    List all CSV tables from the given dataset folder.
    Returns table names without the .csv extension.
    """
    folder = os.path.join(FILES_ROOT, dataset_name)
    return [f.replace('.csv', '') for f in os.listdir(folder) if f.endswith('.csv')]

# === 📊 Chart Rendering === #

# --- Function: plot_chart ---
def plot_chart(df, chart_type, x_col, y_col=None):
    """
    Generate a matplotlib figure from the given DataFrame and chart parameters.

    Args:
        df (pd.DataFrame): The table data.
        chart_type (str): Type of chart (Bar, Line, Pie, Scatter).
        x_col (str): Column to use for x-axis.
        y_col (str, optional): Column for y-axis (not required for Pie chart).

    Returns:
        fig (matplotlib.figure.Figure): The plotted chart figure.
    """
    try:
        fig, ax = plt.subplots(figsize=(10, 5))

        if chart_type == 'Bar Chart':
            df.groupby(x_col)[y_col].sum().plot(kind='bar', ax=ax)
        elif chart_type == 'Line Chart':
            df.plot(x=x_col, y=y_col, kind='line', ax=ax)
        elif chart_type == 'Pie Chart':
            df.groupby(x_col)[y_col].sum().plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_ylabel('')
        elif chart_type == 'Scatter Plot':
            df.plot(kind='scatter', x=x_col, y=y_col, ax=ax)
        else:
            raise ValueError("Unsupported chart type")

        ax.set_title(f"{chart_type}: {x_col} vs {y_col}")
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        return fig

    except Exception as e:
        raise RuntimeError(f"❌ Failed to generate chart: {e}")
