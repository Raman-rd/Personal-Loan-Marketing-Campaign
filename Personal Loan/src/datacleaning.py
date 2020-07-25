import pandas as pd
import numpy as np


if __name__=="__main__":
    
    df = pd.read_excel("input\debarun_corr.xlsx",sheet_name="Data")


    df = df.drop("ID",axis = 1)

    df["Income"] = np.where(df["Income"] > 150,150,df["Income"])

    zip_map_mort=df.groupby("ZIP Code").max()["Mortgage"].to_dict()
    
    df["ZIP Code"] = df["ZIP Code"].map(zip_map_mort)

    df.CCAvg = np.where(df.CCAvg>5,5,df.CCAvg)

    df["Mortgage"] = np.where(df["Mortgage"]==0,0,1)

    df.to_csv("input\cleaned_data.csv",index=False)


   

