from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from data import getData, preprocessData
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from network import DeepLearningClassifier
from evaluate import evaulateModel
import hydra



logger.add("{time:YYYY-MM-DD}.log")
    

def getConfig(cfg):
    path_csv = cfg.paths.path_csv
    epochs = cfg.network.epochs
    batch_size = cfg.network.batch_size
    learning_rate = cfg.network.learning_rate
    logger.info(f"CSV Path: {path_csv}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Learning Rate: {learning_rate}")
    return path_csv, epochs, batch_size, learning_rate

@hydra.main(config_path="../conf", config_name="config.yaml")
def main(cfg):
    # 1. Get Config Data and input file
    path_csv, epochs, batch_size, learning_rate = getConfig(cfg)
    path_parquet = "https://s3.amazonaws.com/BUCKET_FOR_FILE_TRANSFER/ml_dataset.parquet"

    # 2. Get input and output data
    input, output = getData(path_parquet, path_csv)
    logger.info(f"Input Shape: {input.shape}")
    logger.info(f"Output Length: {len(output)}")

    # 3. Presprocess data 
    X_train, y_train, X_test, y_test = preprocessData(input, output)

    # 4. Run Data through training algorithm
    trainedModel = DeepLearningClassifier(X_train, y_train,  epochs, batch_size, learning_rate)

    # 5. Evaluate Model
    evaulateModel(trainedModel, X_test, y_test)

if __name__ == "__main__":
    main()




