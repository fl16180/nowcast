import pandas as pd


def save_csv(df, filename):
    """ Shortcut for cleanly saving prediction dataframe to file """
    df.reset_index().to_csv(filename, index=False)

