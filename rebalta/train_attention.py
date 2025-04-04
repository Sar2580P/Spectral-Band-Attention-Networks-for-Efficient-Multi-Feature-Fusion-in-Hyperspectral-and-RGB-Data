import argparse
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.covariance import EllipticEnvelope
from tqdm import tqdm

from rebalta.arguments import arguments, Arguments
from rebalta.datasets.generate_datasets import get_loader_function, \
    produce_splits
from rebalta.models.model_2 import Model2
from rebalta.models.model_3 import Model3
from rebalta.models.model_4 import Model4


def train_network(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, model,
                  args: Arguments):
    """
    Train and validate model.

    :param x_train: Samples for training.
    :param y_train: Labels for training.
    :param x_val: Samples for evaluation.
    :param y_val: Labels for evaluation.
    :param model: Model designed for training.
    :param args: Parsed arguments.
    :return: None.
    """
    training_history = []
    validation_history = []
    loss_history = []
    training_time = []
    validation_time = []
    for current_epoch in range(args.epochs):
        torch.enable_grad()
        model.train()
        for param in model.parameters():
            param.requires_grad = True

        print("Epoch {}:".format(current_epoch))
        print("\tTraining phase:")

        training_accuracies = []
        losses = []
        begin = time.time()

        x_train_list = [x_train[i:i + args.batch_size] for i in range(0, len(x_train), args.batch_size)]
        y_train_list = [y_train[i:i + args.batch_size] for i in range(0, len(y_train), args.batch_size)]

        for x, y in tqdm(zip(x_train_list, y_train_list), total=len(x_train_list)):
            x = torch.from_numpy(x.astype("float32")).unsqueeze(1).type(torch.cuda.FloatTensor)
            y = torch.from_numpy(y.astype("float32")).type(torch.cuda.FloatTensor)
            model.zero_grad()
            model.optimizer.zero_grad()
            out = model(x, y, infer=False)
            loss = model.loss(out, y)
            losses.append(loss.clone().detach().cpu().numpy())
            accuracy = (torch.argmax(out, dim=1) == torch.argmax(y, dim=1)).sum().type(torch.cuda.DoubleTensor) / \
                       y.shape[0]
            training_accuracies.append(accuracy.cpu().numpy())
            loss.backward()
            model.optimizer.step()

        training_time.append(time.time() - begin)

        validation_accuracies = []

        torch.no_grad()
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        print("\tValidation phase:")
        begin = time.time()

        x_val_list = [x_val[i:i + args.batch_size] for i in range(0, len(x_val), args.batch_size)]
        y_val_list = [y_val[i:i + args.batch_size] for i in range(0, len(y_val), args.batch_size)]

        for x, y in tqdm(zip(x_val_list, y_val_list), total=len(x_val_list)):
            x = torch.from_numpy(x.astype("float32")).unsqueeze(1).type(torch.cuda.FloatTensor)
            y = torch.from_numpy(y.astype("int32")).type(torch.cuda.LongTensor)
            accuracy = (torch.argmax(model(x, y, infer=False), dim=1) == torch.argmax(y, dim=1)).sum().type(
                torch.cuda.DoubleTensor) / y.shape[0]
            validation_accuracies.append(accuracy.cpu().numpy())

        validation_time.append(time.time() - begin)
        loss_history.append(np.mean(losses))
        training_history.append(np.mean(training_accuracies))
        validation_history.append(np.mean(validation_accuracies))
        print("\tLoss: {0}% ".format(loss_history[-1]))
        print("\tTraining accuracy: {0}% ".format(training_history[-1]))
        print("\tValidation accuracy: {0}% ".format(validation_history[-1]))
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        pickle.dump(training_history,
                    open(os.path.join(args.output_dir, args.run_idx + "_training_history.pkl"), "wb"))
        pickle.dump(validation_history,
                    open(os.path.join(args.output_dir, args.run_idx + "_validation_history.pkl"), "wb"))
        pickle.dump(loss_history, open(os.path.join(args.output_dir, args.run_idx + "_loss_history.pkl"), "wb"))
        pickle.dump(training_time,
                    open(os.path.join(args.output_dir, args.run_idx + "_time_training.pkl"), "wb"))
        pickle.dump(validation_time,
                    open(os.path.join(args.output_dir, args.run_idx + "_time_validation.pkl"), "wb"))
        if current_epoch == 0 or (validation_history[-1] > max(validation_history[:-1])):
            print("\tSaving model...")
            torch.save(model, os.path.join(args.output_dir, args.run_idx + "_model.pt"))
        if current_epoch > args.patience:
            if max(validation_history[:-args.patience]) > max(validation_history[-args.patience:]):
                print("\tBail...")
                break


def infer_network(x_test: np.ndarray, y_test: np.ndarray, args: Arguments, input_size: int):
    """
    Conduct inference on the trained model.

    :param x_test: Samples for testing.
    :param y_test: Labels for testing.
    :param args: Parsed arguments.
    :param input_size: Size of the initial band spectrum.
    :return: None.
    """
    model = torch.load(os.path.join(args.output_dir, args.run_idx + "_model.pt"))
    testing_accuracies = []

    torch.no_grad()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print("\tTesting:")

    begin = time.time()
    x_test_list = [x_test[i:i + args.batch_size] for i in range(0, len(x_test), args.batch_size)]
    y_test_list = [y_test[i:i + args.batch_size] for i in range(0, len(y_test), args.batch_size)]

    for x, y in tqdm(zip(x_test_list, y_test_list), total=len(x_test_list)):
        x = torch.from_numpy(x.astype("float32")).unsqueeze(1).type(torch.cuda.FloatTensor)
        y = torch.from_numpy(y.astype("int32")).type(torch.cuda.LongTensor)
        accuracy = (torch.argmax(model(x, y, infer=True), dim=1) == torch.argmax(y, dim=1)).sum().type(
            torch.cuda.DoubleTensor) / y.shape[0]
        testing_accuracies.append(accuracy.cpu().numpy())

    testing_time = time.time() - begin
    testing_accuracy = [np.mean(testing_accuracies)]
    print("\tTesting accuracy: {}% ".format(testing_accuracy[0]))
    if model.uses_attention:
        heatmaps_per_class = model.get_heatmaps(input_size=input_size)
        pickle.dump(heatmaps_per_class,
                    open(os.path.join(args.output_dir, args.run_idx + "_attention_bands.pkl"), "wb"))

    pickle.dump(testing_accuracy,
                open(os.path.join(args.output_dir, args.run_idx + "_testing_accuracy.pkl"), "wb"))
    pickle.dump(testing_time, open(os.path.join(args.output_dir, args.run_idx + "_time_testing.pkl"), "wb"))


def load_model(n_attention_modules: int, n_classes: int, input_dimension: int, uses_attention: bool) -> torch.nn.Module:
    """
    Load attention-based model architectures.

    :param n_attention_modules: Number of attention modules, i.e. {2, 3, 4}.
    :param n_classes: Number of classes.
    :param input_dimension: Size of the spectrum channel.
    :param uses_attention: Boolean variable indicating whether to use attention.
    :return: Instance of the model.
    """
    if n_attention_modules == 2:
        return Model2(num_of_classes=n_classes, input_dimension=input_dimension, uses_attention=uses_attention)
    elif n_attention_modules == 3:
        return Model3(num_of_classes=n_classes, input_dimension=input_dimension, uses_attention=uses_attention)
    elif n_attention_modules == 4:
        return Model4(num_of_classes=n_classes, input_dimension=input_dimension, uses_attention=uses_attention)
    raise argparse.ArgumentError(argument=None,
                                 message="Incorrect number of attention modules.\n"
                                         "Possible choices: [2, 3, 4]\n"
                                         "Passed argument: {}".format(n_attention_modules))


def run(args: Arguments, selected_bands: np.ndarray = None):
    """
    Method for running the experiments.

    :param args: Parsed arguments.
    :param selected_bands: Bands selected by the outlier detection algorithm.
    :return: None.
    """
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    os.makedirs(args.output_dir, exist_ok=True)
    print("Training model for dataset: {}".format(os.path.basename(os.path.normpath(args.dataset_path))))
    samples, labels = get_loader_function(data_path=args.dataset_path, ref_map_path=args.labels_path)
    if args.selected_bands is not None:
        selected_bands = np.loadtxt(fname=args.selected_bands).astype(int)
    if selected_bands is not None:
        samples = samples[..., selected_bands]
        print("Selected bands: {}".format(selected_bands))
        print("Number of selected bands: {}".format(samples.shape[-1]))
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = produce_splits(samples=samples,
                                                                          labels=labels,
                                                                          validation_size=args.validation,
                                                                          test_size=args.test)
    try:
        model = load_model(n_attention_modules=args.modules,
                           n_classes=int(labels.max() + 1),
                           input_dimension=x_train.shape[-1],
                           uses_attention=str2bool(args.attn))
    except argparse.ArgumentError as e:
        print(e)
        sys.exit("Incorrect arguments specification.")
    model.to(device)
    train_network(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, model=model, args=args)
    infer_network(x_test=x_test, y_test=y_test, args=args, input_size=x_train.shape[-1])


def plot_heatmaps(heatmaps: np.ndarray, args: Arguments, show_fig: bool):
    """
    Plot heatmaps for each class.

    :param heatmaps: Array containing attention scores for all classes.
    :param args: Arguments.
    :param show_fig: Boolean indicating whether to show the selected bands heatmap.
    :return: None.
    """
    fig, axis = plt.subplots()
    heatmap = axis.pcolor(heatmaps)
    axis.set_yticklabels(list(range(heatmaps.shape[0])), minor=True)
    plt.colorbar(heatmap)
    fig.set_size_inches(12, 4)
    plt.title("Attention heatmaps scores")
    plt.ylabel("Class")
    plt.xlabel("Band")
    plt.savefig(os.path.join(args.output_dir, args.run_idx + "_attention_map.pdf"))
    if show_fig:
        plt.show()


def eval_heatmaps(args: Arguments) -> np.ndarray:
    """
    Detect outliers in the collected heatmaps over all classes.

    :param args: Parsed arguments.
    :return: Array containing selected bands.
    """
    heatmaps = pickle.load(open(os.path.join(args.output_dir, args.run_idx + "_attention_bands.pkl"), "rb"))
    plot_heatmaps(heatmaps=heatmaps, args=args, show_fig=True)
    clf = EllipticEnvelope(contamination=float(args.cont))
    heatmaps = np.expand_dims(np.mean(heatmaps, axis=0), axis=1)
    outliers = clf.fit(heatmaps).predict(heatmaps)
    outliers[outliers == 1] = 0
    nonzero = np.nonzero(outliers)
    selected_bands = np.unique(nonzero)
    print("Selected bands: {0}".format(selected_bands))
    np.savetxt(os.path.join(args.output_dir,
                            args.run_idx + "_selected_bands"),
               selected_bands,
               delimiter="\n",
               fmt="%d")
    return selected_bands


def str2bool(string_arg: str) -> bool:
    """
    Parse string argument to bool.

    :param string_arg: Argument indicating whether to use attention mechanism.
    :return: Parsed boolean value indicating whether to use attention.
    """
    if string_arg.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif string_arg.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main(args: Arguments):
    """
    Run band selection algorithm and then train model on selected bands.

    :param args: Parsed arguments.
    :return: None.
    """
    try:
        if not str2bool(args.attn):
            # If model was not using attention, train and evaluate on data without selecting bands.
            args = args._replace(run_idx=args.run_idx + "_no_attention")
        if args.selected_bands is not None and str2bool(args.attn):
            # Assert that there will be no band selection on the reduced dataset.
            raise argparse.ArgumentError(argument=None,
                                         message="Cannot perform band selection on the reduced dataset.")
    except (argparse.ArgumentTypeError, argparse.ArgumentError) as e:
        print(e)
        sys.exit("Incorrect arguments specification.")
    run(args)
    try:
        if str2bool(args.attn):
            # If model was using attention, select bands from obtained heatmap.
            selected_bands = eval_heatmaps(args)
            # After selection train new model without attention on reduced data.
            args = args._replace(attn="false")
            args = args._replace(run_idx=args.run_idx + "_no_attention")
            run(args, selected_bands)
    except argparse.ArgumentTypeError as e:
        print(e)
        sys.exit("Incorrect arguments specification.")


if __name__ == "__main__":
    parsed_args = arguments()
    main(args=parsed_args)