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
# 1. Define Paths and Filenames
path_parquet = "https://s3.amazonaws.com/BUCKET_FOR_FILE_TRANSFER/ml_dataset.parquet"
path_csv = "ml_dataset.csv"

# 2. Get input and output data
input, output = getData(path_parquet, path_csv)
logger.info(f"Input Shape: {input.shape}")
logger.info(f"Output Length: {len(output)}")

# 3. Presprocess data
X_train, y_train, X_test, y_test = preprocessData(input, output)

# 3. Run Data through training algorithm
trainedModel = DeepLearningClassifier(X_train, y_train, epochs=10, batch_size=64, learning_rate=0.001)

# 4. Evaluate Model
evaulateModel(trainedModel, X_test, y_test)




