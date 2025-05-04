import pandas as pd


def load_to_df(file_path: str) -> pd.DataFrame:
    data = pd.read_excel(
        file_path, 
        header=None, 
        usecols=[0, 7, 11, 12, 13], 
        names=["player", "result", "planets_combo", "intensity_degree", "planets_intensity"],
        engine='openpyxl'
    )

    return data