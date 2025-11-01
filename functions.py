import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def histogram_boxplot(data, feature, figsize=(12, 7), kde=False, bins=None, dropna=True):
    """
    Boxplot and histogram combined with key statistics displayed and dynamic bin calculation.

    Parameters:
    - data: pandas DataFrame
    - feature: column name to visualize
    - figsize: tuple, size of figure (default (12,7))
    - kde: whether to show density curve (default False)
    - bins: number of bins for histogram (default None — auto-calculated)
    - dropna: whether to drop NaN values before plotting (default True)
    """

    # Optionally drop NaN values
    series = data[feature].dropna() if dropna else data[feature]

    # Handle case where column might be empty
    if series.isnull().all():
        raise ValueError(f"Column '{feature}' contains only NaN values. Nothing to plot.")

    # Calculate key statistics
    mean = series.mean()
    median = series.median()
    std = series.std()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    # Auto-calculate bins using Freedman–Diaconis rule if not provided
    if bins is None:
        n = len(series)
        if n > 1:
            bin_width = 2 * iqr / np.cbrt(n)
            if bin_width > 0:
                bins = int(np.ceil((series.max() - series.min()) / bin_width))
            else:
                bins = 10  # fallback if IQR=0
        else:
            bins = 10  # fallback if only one data point

    # Create figure with subplots
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize
    )

    # Boxplot
    sns.boxplot(
        x=series, ax=ax_box2, showmeans=True, color="violet"
    )

    # Histogram
    sns.histplot(
        x=series, kde=kde, ax=ax_hist2, bins=bins, color="skyblue"
    )

    # Vertical lines for key stats
    ax_hist2.axvline(mean, color="green", linestyle="--", label=f"Mean = {mean:.2f}")
    ax_hist2.axvline(median, color="black", linestyle="-", label=f"Median = {median:.2f}")
    ax_hist2.axvline(q1, color="orange", linestyle=":", label=f"Q1 = {q1:.2f}")
    ax_hist2.axvline(q3, color="red", linestyle=":", label=f"Q3 = {q3:.2f}")

    # Text annotation box with key statistics
    stats_text = (
        f"Mean: {mean:.2f}\n"
        f"Median: {median:.2f}\n"
        f"Std Dev: {std:.2f}\n"
        f"Q1: {q1:.2f}\n"
        f"Q3: {q3:.2f}\n"
        f"IQR: {iqr:.2f}\n"
        f"Bins: {bins}"
    )

    ax_hist2.text(
        0.98, 0.95, stats_text,
        transform=ax_hist2.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8)
    )

    # Add legend and tidy up
    ax_hist2.legend()
    plt.tight_layout()
    plt.show()


# function to create labeled barplots

def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


# function to plot stacked bar chart

def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 1, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


### Function to plot distributions

def distribution_plot_wrt_target(data, predictor, target):

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()


###comparative boxplots

def comparative_boxplot(data, x_col, y_col, title="", x_label="", y_label="", palette="Oranges"):
    """
    Creates a comparative boxplot between a categorical and a numerical variable.

    Parameters:
        data (DataFrame): dataset to use.
        x_col (str): name of the categorical column (x-axis).
        y_col (str): name of the numerical column (y-axis).
        title (str): plot title.
        x_label (str): label for the x-axis.
        y_label (str): label for the y-axis.
        palette (str or list): Seaborn color palette.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=x_col, y=y_col, palette='Purples')
    plt.title(title, fontsize=14)
    plt.xlabel(x_label if x_label else x_col, fontsize=12)
    plt.ylabel(y_label if y_label else y_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


##comparative barplots

def comparative_barplot(data, x_col, y_col, title="", x_label="", y_label="", palette="Purples"):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x=x_col, y=y_col, palette=palette, errorbar="sd")
    plt.title(title, fontsize=14)
    plt.xlabel(x_label if x_label else x_col, fontsize=12)
    plt.ylabel(y_label if y_label else y_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



### outlier detector

def detect_outliers_iqr(data, threshold=0.01):
    outlier_summary = []
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Identify outliers
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
        n_outliers = len(outliers)
        pct_outliers = 100 * n_outliers / len(data)
        # Print summary
        print(f"Column: {col} - Number of Outliers: {n_outliers}")
        print(f"Column: {col} - % of Outliers: {pct_outliers:.2f}%\n")
        # Add to list if exceeds threshold
        if pct_outliers > threshold * 100:
            outlier_summary.append((col, n_outliers, pct_outliers))
        # Plot boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=data[col], color='orange')
        sns.stripplot(x=outliers, color='red', size=4, label='Outliers')
        plt.title(f'Boxplot with Outliers for {col}', fontsize=14)
        plt.legend()
        plt.show()
    return pd.DataFrame(outlier_summary, columns=['Column', 'Num_Outliers', 'Pct_Outliers'])