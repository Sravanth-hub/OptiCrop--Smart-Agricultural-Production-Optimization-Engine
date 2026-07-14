import pickle
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    try:
        y_prob = model.predict_proba(X_test)
        roc_auc = roc_auc_score(
            y_test,
            y_prob,
            multi_class="ovr",
            average="macro"
        )
        print("\nROC-AUC Score:", roc_auc)
    except:
        print("\nROC-AUC Score cannot be calculated.")

    # Save model
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "model.pkl")

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    print("\n✅ Model saved successfully:", model_path)