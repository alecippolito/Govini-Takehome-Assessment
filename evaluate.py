from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
import torch 

def evaulateModel(model, X_test, y_test):
    # 1. Make predictions on test data with trained model
    with torch.no_grad():
        model.eval()
        y_pred_probs = model(X_test)
        y_pred = (y_pred_probs > 0.5).float()

    # 2. Convert pred to numpy
    y_pred_np = y_pred.numpy()
    y_test_np = y_test.numpy()

    # 3. Evaluate accuracy
    accuracy = accuracy_score(y_test_np, y_pred_np)
    logger.info(f"Accuracy: {accuracy}")
