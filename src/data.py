import csv
import os
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from encode import encodeStrings, encodeUCIDs, encodeCodes, encodeCPCs

def getData(path_parquet, path_csv):
    logger.info("Getting Data...")
    # 1. Read in parquet and convert it to csv for easier data handling
    logger.info(f"Reading Parquet {path_parquet} and writing data to csv {path_csv}")
    if not os.path.isfile(path_csv): 
        data_parquet = pd.read_parquet(path_parquet)
        data_parquet.to_csv(path_csv, index=False)

    # 2. Read data from csv
    df = pd.read_csv(path_csv)

    logger.info(f"First 5 rows of the dataset:\n{df.head()}")

    # 3. Get data from columns and make ndarrays
    titles = np.array(df['title'].values)
    abstracts = np.array(df['abstract'].values)
    ucids = np.array(df['ucid'].values)
    codes = np.array(df['code'].values)
    output = df['labels'].values

    logger.info("Encoding data...")
    # 5. Encode column data
    logger.info("Encoding titles...")
    titlesEncoded = encodeStrings(titles)
    logger.info("Encoding abstracts...")
    abstractsEncoded = encodeStrings(abstracts)
    logger.info("Encoding ucids...")
    ucidsEncoded = encodeUCIDs(ucids)
    logger.info("Encoding codes...")
    codesEncoded = encodeCodes(codes)
    logger.info("...Done encoding data")

    # 6. logger.info out information on encoded data
    logger.info(f"Titles Data Information: Rows: {len(titlesEncoded)}, Columns: {len(titlesEncoded[0])}")
    logger.info(f"Abstracts Data Information: Rows: {len(abstractsEncoded)}, Columns: {len(abstractsEncoded[0])}")
    logger.info(f"UCID Data Information: Rows: {len(ucidsEncoded)}, Columns: {len(ucidsEncoded[0])}")
    logger.info(f"Codes Data Information: Rows: {len(codesEncoded)}, Columns: {len(codesEncoded[0])}")

    # 7. Concat all input data into one 2d array
    input = np.concatenate((titlesEncoded, abstractsEncoded, ucidsEncoded, codesEncoded), axis=1)

    # 8. Return input and output data
    return input, output 

# Preprocessing Data Function
def preprocessData(X, y):
    logger.info("Staring preprocessing of data...")
    # 1. Split into Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Standardize input data and convert to torch tensor
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    # 3. Encode labels to numbers (since I didn't encode labels) and convert to torch tensor
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)

    logger.info("...Done preprocessing data - split data into training and testing set")
    # 4. Return Processed Data: X_train, y_train, X_test, y_test
    return X_train, y_train, X_test, y_test