import logging
import sys, io
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score


def collect_metrics(y_pred, y_true):
    #logging.debug(f"y_pred {y_pred}")
    #logging.debug(f"y_true {y_true}")
    accuracy = accuracy_score(y_true, y_pred)

    # Print classification report
    logging.debug("Classification Report:")
    logging.debug(classification_report(y_true, y_pred))
    logging.debug(f"{classification_report(y_true, y_pred)}")

    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logging.debug(confusion_matrix(y_true, y_pred))
    true_negative, false_positive, false_negative, true_positive = cm.ravel()

    # Calculate sensitivity (true positive rate or recall)
    sensitivity = true_positive / (true_positive + false_negative)

    # Calculate sensitivity (true positive rate or recall)
    sensitivity = true_positive / (true_positive + false_negative)

    # Calculate specificity
    specificity = true_negative / (true_negative + false_positive)

    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    logging.debug("Confusion Matrix:")
    logging.debug(cm)
    logging.debug(f"Accuracy: {accuracy}")
    logging.debug(f"Sensitivity: {sensitivity}")
    logging.debug(f"Specificity: {specificity}")
    logging.debug(f"Precision: {precision}")
    logging.debug(f"Recall: {recall}")
    logging.debug(f"F1: {f1}")

    logging.debug(f"AUC: {auc}")


def summary(model):
    # Redirect stdout to capture the summary
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()

    # Print the model summary (it will be captured in the StringIO buffer)
    model.summary()

    # Get the captured summary
    model_summary = sys.stdout.getvalue()

    # Restore stdout to its original value
    sys.stdout = original_stdout

    # Log the model summary
    logging.debug("Model Summary:\n%s", model_summary)