import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import pickle
from sklearn.metrics import accuracy_score
import os
import joblib
from processing.utils import read_yaml
import glob

def load_data(data_dir):
    tr_preds = glob.glob(os.path.join(data_dir, 'rgb_*_train.npy'))
    pattern = f"hsi_*_train_{hsi_config['data_type']}__{hsi_config['preprocessing']}__{hsi_config['C']}.npy"
    tr_preds += glob.glob(os.path.join(data_dir, pattern))

    val_preds = [file.replace('train', 'val') for file in tr_preds]
    train_data = join_base_preds(tr_preds)
    val_data = join_base_preds(val_preds)
    return train_data, val_data

def join_base_preds(pred_files):
    X , Y = None, None

    for file in pred_files:
        data = np.load(file)
        if X is None:
            X = data[:, :-1]
        else:
            X = np.concatenate((X, data[: ,:-1]), axis=1)
        Y = data[:, -1] if Y is None else Y

    assert X.shape[0]==Y.shape[0], "Number of samples should be the same"
    return np.concatenate((X, Y.reshape(-1, 1)), axis=1)



def prepare_data(data, num_classes):
    X = data[:, :-1]  # All columns except the last one (model predictions)
    y = data[:, -1]   # Last column as the true class label
    num_models = X.shape[1] // num_classes
    X = X.reshape(-1, num_models, num_classes)  # Reshape to (num_samples, num_models, num_classes)
    return X, y

def train_ensemble(X_train, y_train, num_classes, models):
    num_models = X_train.shape[1]
    X_train_reshaped = X_train.reshape(-1, num_models * num_classes)  # Flatten the predictions for all models

    best_model = None
    best_acc = 0
    regressed_weights = None
    model_results = {}

    # Try different regression models
    for model in models:
        model_name = model.__class__.__name__
        print(f"Training with {model_name}...")

        model.fit(X_train_reshaped, y_train)

        # Extract weights or feature importance
        if hasattr(model, 'coef_'):
            weights = model.coef_.reshape
        elif hasattr(model, 'feature_importances_'):
            weights = model.feature_importances_
        else:
            weights = np.ones((num_models, num_classes))  # Fallback in case weights aren't available

        # Predict on the training set
        # y_train_pred = predict_with_weights(X_train, weights)
        y_train_pred = model.predict(X_train_reshaped)
        acc = accuracy_score(y_train, y_train_pred)

        print(f"Accuracy with {model_name}: {acc}")

        # Store results for this model
        model_results[model_name] = {
            model_name : {'weights': weights,
                        'accuracy': acc ,
                        'model' : model}
        }

        joblib.dump(model, f'{model_name}_model.pkl')

        # Update the best model if current one is better
        if acc > best_acc:
            best_acc = acc
            best_model = model
            regressed_weights = weights

    return best_model, regressed_weights, model_results

def predict_with_weights(X, weights):
    weighted_sum = np.einsum('ijk,jk->ik', X, weights)  # Weighted sum of the model logits
    y_pred = np.argmax(weighted_sum, axis=1)  # Argmax to get the predicted class
    return y_pred

def evaluate_ensemble(X_val, y_val, model):
    y_val_pred = model.predict(X_val.reshape(-1, X_val.shape[1] * X_val.shape[2]))
    acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {acc}")
    return y_val_pred, acc

def save_results(y_true, y_pred, file_path):
    np.save(file_path, {'y_true': y_true, 'y_pred': y_pred})


def save_model_results(model_results, pkl_file_path):
    with open(pkl_file_path, 'wb') as pkl_file:
        pickle.dump(model_results, pkl_file)


def main(data_dir, num_classes, save_file, json_file_path):
    train_data, val_data = load_data(data_dir)

    X_train, y_train = prepare_data(train_data, num_classes)
    X_val, y_val = prepare_data(val_data, num_classes)



    models = [
        # LogisticRegression(max_iter=5000, multi_class='multinomial'),
        SVC(),  # Note: SVC is used for classification, while SVR is used for regression
        # xgb.XGBClassifier()  # Use XGBClassifier for classification tasks
    ]

    best_model, weights, model_results = train_ensemble(X_train, y_train, num_classes, models)
    best_model = joblib.load("SVC_model.pkl")
    print("Evaluating on validation data...")
    y_val_pred, val_acc = evaluate_ensemble(X_val, y_val, best_model)

    print(f"Saving results to {save_file}...")
    save_results(y_val, y_val_pred, save_file)

    print(f"Saving model results to {json_file_path}...")
    save_model_results(model_results, json_file_path)

if __name__ == "__main__":
    hsi_config = read_yaml('models/hsi/config.yaml')

    SOURCE_DIR = f"results/ensemble/base_models/classes-{hsi_config['num_classes']}"
    assert os.path.exists(SOURCE_DIR), f"Source directory {SOURCE_DIR} does not exist"

    BASE_DIR = f"results/ensemble/regression/classes-{hsi_config['num_classes']}"
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)

    SAVE_DIR = os.path.join(BASE_DIR, f"fold_{hsi_config['fold']}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    data_dir = os.path.join(SOURCE_DIR , f"fold_{hsi_config['fold']}")

    prefix_name = f"RegressionEnsemble__{hsi_config['data_type']}__{hsi_config['preprocessing']}__{hsi_config['C']}"
    prediction_save_path = f"{SAVE_DIR}/{prefix_name}__predictions.npy"  # File to save y_true, y_pred, and weights
    model_metadata_save_path = f"{SAVE_DIR}/{prefix_name}__model_performance.pkl"  # File to save weights and accuracy of each model

    main(data_dir, hsi_config['num_classes'], prediction_save_path, model_metadata_save_path)