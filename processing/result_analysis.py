import pickle
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Load predictions from the pickle file
file_path = "results/hsi/classes-96/fold-1/evaluations/densenet-densenet_mini__original__none__predictions.pkl"

with open(file_path, 'rb') as f:
    predictions = pickle.load(f)

y_hat = predictions['y_hat']
y_true = predictions['y_true']

y_pred = []
y_ground = []
y_pred_prob = []  # To store the probability of the predicted class

for i, tensor in enumerate(y_hat):
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(tensor, dim=1)

    # Get the predicted class and its probability
    class_pred = torch.argmax(probabilities, dim=1)
    class_pred_prob = torch.max(probabilities, dim=1).values

    y_pred.extend(class_pred.tolist())
    y_ground.extend(y_true[i].tolist())
    y_pred_prob.extend(class_pred_prob.tolist())

num_classes = predictions['training_config']['num_classes']

def per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    return per_class_acc, cm

def classes_below_threshold(y_true, y_pred, threshold=0.85):
    per_class_acc, cm = per_class_accuracy(y_true, y_pred)
    below_threshold_classes = {i: acc for i, acc in enumerate(per_class_acc) if acc < threshold}
    return below_threshold_classes, cm

def get_probability_distributions(y_true, y_pred, y_pred_prob, num_classes):
    # Initialize a dictionary to hold the probability distributions
    prob_distributions = {i: {j: [] for j in range(num_classes)} for i in range(num_classes)}

    # Fill the dictionary with probabilities
    for true_class, pred_class, pred_prob in zip(y_true, y_pred, y_pred_prob):
        prob_distributions[true_class][pred_class].append(pred_prob)

    # Calculate mean and median for each class
    mean_median_distributions = {i: {j: {'mean': np.mean(prob_distributions[i][j]) if prob_distributions[i][j] else 0,
                                         'median': np.median(prob_distributions[i][j]) if prob_distributions[i][j] else 0}
                                      for j in range(num_classes)} for i in range(num_classes)}

    return mean_median_distributions

def plot_misclassification_histograms(cm, low_accuracy_classes, mean_median_distributions, num_classes, save_path="misclassification_histograms.png"):
    classes = list(low_accuracy_classes.keys())
    fig, axes = plt.subplots(nrows=len(classes), ncols=1, figsize=(15, 6 * len(classes)))

    if len(classes) == 1:
        axes = [axes]

    for idx, class_idx in enumerate(classes):
        ax = axes[idx]
        misclassified_counts = cm[class_idx, :]
        correct_class_count = misclassified_counts[class_idx]

        # Zero out the correct class to focus on misclassifications
        misclassified_counts[class_idx] = 0

        bars = ax.bar(range(num_classes), misclassified_counts, color='skyblue')
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(range(num_classes), rotation=90)
        ax.set_title(f'Misclassifications for Class {class_idx} (class accuracy: {(low_accuracy_classes[class_idx]*100):.2f}%)')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Count')

        # Add mean and median probability on top of each bar
        for bar, class_id in zip(bars, range(num_classes)):
            height = bar.get_height()
            if height == 0:
                continue
            mean_prob = mean_median_distributions[class_idx][class_id]['mean']
            median_prob = mean_median_distributions[class_idx][class_id]['median']
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'Mean: {mean_prob:.2f}\nMedian: {median_prob:.2f}',
                    ha='center', va='bottom', rotation=90)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Get classes with accuracy below 0.85 and confusion matrix
low_accuracy_classes, cm = classes_below_threshold(y_ground, y_pred, threshold=0.85)

# Get the mean and median probability distributions
mean_median_distributions = get_probability_distributions(y_ground, y_pred, y_pred_prob, num_classes)

# Print the low accuracy classes and their accuracies
print(f'Classes with accuracy below 0.85: {low_accuracy_classes}')

# Plot and save the confusion matrix for the low accuracy classes
if low_accuracy_classes:
    plot_misclassification_histograms(cm, low_accuracy_classes, mean_median_distributions, num_classes, save_path="low_accuracy_confusion_matrix.png")
    print("Confusion matrix saved as 'low_accuracy_confusion_matrix.png'")
else:
    print("No classes found with accuracy below the threshold.")
