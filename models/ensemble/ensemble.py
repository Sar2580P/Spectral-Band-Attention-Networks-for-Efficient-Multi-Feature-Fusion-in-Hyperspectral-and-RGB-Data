from tqdm import tqdm
import sys
sys.path.append('Preprocessing')
sys.path.append('models')
from train_eval import *
from models.model_architectures import *
from modules import *
from dynamic_weighting import *
from utils import *
import pickle
import time
torch.set_float32_matmul_precision('high')


def get_fold(fold:int)->dict:
  rgb_dir, hsi_dir = 'models/rgb' , 'models/hsi'
  rgb_models = ['densenet', 'google_net' , 'resnet-34' , 'resnet-50']
  hsi_models = ['dense_net']
  res = {}

  for dir, modality, models in zip([rgb_dir , hsi_dir] , ['rgb', 'hsi'] , [rgb_models , hsi_models]):
    res[modality] = {}
    for model in models:
      files = os.listdir(os.path.join(dir , model , 'evaluations'))
      x = [1 if file.split('__')[-2][-1]==str(fold) else 0 for file in files]

      res[modality][model] = files[np.argmax(x)]

  return res


def get_logit_df():
  for fold in tqdm(range(5)):
    dict_ = get_fold(fold)

    for modality in dict_.keys():
      for model_name in dict_[modality].keys():
        path = os.path.join('models', modality, model_name , 'evaluations', dict_[modality][model_name])
        file = pickle.load(open(path, 'rb'))
        y_hat, y_true = file['y_hat'] , file['y_true']

        result = pd.DataFrame(columns = [str(i) for i in range(98)])
        labels = []
        for batch in y_true:
          labels.extend(list(batch.detach().cpu().numpy()))

        for batch in y_hat:
          batch = batch.detach().cpu().numpy()
          for i in range(batch.shape[0]):
            result.loc[len(result)] = batch[i]

        result['labels'] = labels
        result.to_csv(f'models/ensemble/Classifier_Prediction_Data/98/{modality}/{model_name}--{fold}.csv' , index = False)
    time.sleep(2)

  return

if not os.path.exists('models/ensemble/Classifier_Prediction_Data/98'):
  os.mkdir('models/ensemble/Classifier_Prediction_Data/98')
  os.mkdir('models/ensemble/Classifier_Prediction_Data/98/hsi')
  os.mkdir('models/ensemble/Classifier_Prediction_Data/98/rgb')
  get_logit_df()

#_______________________________________________________________________________________________________________________


# for fold in range(5):
#   top = 30    #top 'k' classes
#   p1 = pd.read_csv(f'models/ensemble/Classifier_Prediction_Data/98/hsi_densenet--{fold}.csv')
#   p2 = pd.read_csv(f'models/ensemble/Classifier_Prediction_Data/98/rgb_denseNet_net--{fold}.csv')
#   labels = p1['labels']
#   p1 = p1.iloc[: , :98]
#   p2 = p2.iloc[: , :98]
#   predictions = Gompertz(top=top, argv = (p1, p2))

#   correct = np.where(predictions == labels)[0].shape[0]
#   total = labels.shape[0]
#   acc = correct/total

#   print(f"{fold} : Accuracy =   ",acc*100)
  # classes = []
  # for i in range(p1.shape[1]):
  #     classes.append(str(i+1))

  # metrics(labels,predictions,classes)

#_______________________________________________________________________________________________________________________

def get_ensemble_performance():
  arr = np.arange(0, 1.005, 0.005)

  base_dir = 'models/ensemble/Classifier_Prediction_Data/98'
  rgb_dir , hsi_dir = os.path.join(base_dir,'rgb') , os.path.join(base_dir,'hsi')
  rgb_files , hsi_files = os.listdir(rgb_dir) , os.listdir(hsi_dir)

  for fold in range(5):
    fold_stats = {}
    rgb_fold =  [f for f in rgb_files if f[-5]==str(fold)]
    hsi_fold = [f for f in hsi_files if f[-5]==str(fold)]
    for rgb_file in rgb_fold:
      for hsi_file in hsi_fold:
        p1 = pd.read_csv(os.path.join(rgb_dir, rgb_file))
        p2 = pd.read_csv(os.path.join(hsi_dir , hsi_file))
        labels = p1['labels']
        p1 = p1.iloc[: , :98]
        p2 = p2.iloc[: , :98]

        y_true, y_predicted = labels, None
        accuracies = []
        max = 0
        max_lambda = 0

        for w1 in tqdm(arr):
          preds = []
          for i in range(len(labels)) :
            p = (w1 * p1.iloc[i , :]) + ((1-w1) * p2.iloc[i , :])
            output_class = np.argmax(p, axis =0)
            preds.append(output_class)

          accuracy = np.sum(np.array(preds) == np.array(labels)) / len(labels)
          if accuracy > max:
            max = accuracy
            max_lambda = w1
            y_predicted = preds
          accuracies.append(round(accuracy , 4))

        name = f"hsi-{hsi_file.split('--')[0]}____rgb-{rgb_file.split('--')[0]}"

        fold_stats[name] = {
          'max_accuracy' : max,
          'max_lambda' : max_lambda,
          'y_true' : y_true,
          'y_predicted' : y_predicted,
          'accuracies' : accuracies
        }

    pickle.dump(fold_stats, open(f'models/ensemble/Classifier_Prediction_Data/fold_stats/fold--{fold}.pkl', 'wb'))


if not os.path.exists('models/ensemble/Classifier_Prediction_Data/fold_stats'):
  os.mkdir('models/ensemble/Classifier_Prediction_Data/fold_stats')
  get_ensemble_performance()


def learning_curves():
  BASE_DIR = 'models/ensemble/Classifier_Prediction_Data/varying_classes'
  if not os.path.exists(BASE_DIR):
    os.mkdir(BASE_DIR)

  rgb_dir = 'models/rgb'
  hsi_dir = 'models/hsi'

  rgb_models = ['densenet' , 'google_net' , 'resnet-34' , 'resnet-50']
  hsi_models = ['dense_net']
  ensemble_max_lambda = {'google_net':0.305,
                        'resnet-34':0.45,
                        'resnet-50': 0.45,
                        'densenet': 0.305}

  learning_class_sizes = [12, 24, 37 , 55 , 75]
  results = {}
  for m1 in rgb_models:
    for m2 in hsi_models:
      for class_sz in learning_class_sizes:

        rgb_path = [os.path.join(rgb_dir, m1, 'evaluations', path)
                    for path in os.listdir(os.path.join(rgb_dir, m1, 'evaluations'))
                    if f"var-{class_sz}" in path][0]
        hsi_path =[os.path.join(hsi_dir, m2, 'evaluations', path)
                  for path in os.listdir(os.path.join(hsi_dir, m2, 'evaluations'))
                  if f"var-{class_sz}" in path][0]

        rgb_logits = get_logits('rgb' , m1 , 4 , class_sz, rgb_path)
        hsi_logits = get_logits('hsi' , m2 , 4 , class_sz, hsi_path)
        y_true = rgb_logits['labels']
        rgb_preds , hsi_preds, ensemble_preds = get_pred(rgb_logits , hsi_logits , class_sz , ensemble_max_lambda[m1])

        rgb_acc, hsi_acc , ensemble_acc = get_accuracy(y_true , rgb_preds) , get_accuracy(y_true , hsi_preds) ,get_accuracy(y_true , ensemble_preds)
        print()
        if class_sz not in results.keys():
          results[class_sz] = {}
        if not m1 in results[class_sz].keys():
          results[class_sz][f"rgb_{m1}"] = rgb_acc

        if not m2 in results[class_sz].keys():
          results[class_sz][f"hsi_{m2}"] = hsi_acc
        results[class_sz][f'hsi_{m2}____rgb_{m1}'] = ensemble_acc


  with open(os.path.join(BASE_DIR ,  'learning_curves_with_varying_classes.pkl'), 'wb') as f:
    pickle.dump(results, f)
  return

def get_logits(modality , model_name , fold ,class_sz , path):
  saving_path = f'models/ensemble/Classifier_Prediction_Data/varying_classes/{modality}--{model_name}--fold_{fold}--class_sz_{class_sz}.csv'
  file = pickle.load(open(path, 'rb'))
  y_hat, y_true = file['y_hat'] , file['y_true']
  if not os.path.exists(saving_path):
    file = pickle.load(open(path, 'rb'))
    y_hat, y_true = file['y_hat'] , file['y_true']

    result = pd.DataFrame(columns = [str(i) for i in range(class_sz)])
    labels = []
    for batch in y_true:
      labels.extend(list(batch.detach().cpu().numpy()))

    for batch in y_hat:
      batch = batch.detach().cpu().numpy()
      for i in range(batch.shape[0]):
        result.loc[len(result)] = batch[i]

    result['labels'] = labels
    result.to_csv( saving_path, index = False)
  else :
    result = pd.read_csv(saving_path)
  return result


def get_pred(p1, p2,class_sz, w1):
  p1 = p1.iloc[: , :class_sz]
  p2 = p2.iloc[: , :class_sz]
  ensemble_preds = []
  rgb_preds = []
  hsi_preds = []
  for i in range(len(p1)) :
    p = (w1 * p1.iloc[i , :]) + ((1-w1) * p2.iloc[i , :])
    ensemble_preds.append(np.argmax(p, axis =0))
    rgb_preds.append(np.argmax(p1.iloc[i,:], axis=0))
    hsi_preds.append(np.argmax(p2.iloc[i,:], axis = 0))

  return rgb_preds, hsi_preds, ensemble_preds

def get_accuracy(y_true , y_hat):
  return np.sum(np.array(y_hat) == np.array(y_true)) / len(y_true)


# with open('models/hsi/dense_net/evaluations/hsi_densenet__var-55__fold-4__predictions.pkl', 'rb') as f:
#     rgb_model_stats = pickle.load(f)

# print(rgb_model_stats['y_hat'])

# learning_curves()
#___________________________________________________________________________________________________________________
# name = 'densenet'
# fold = 4
# w1 =  0.305
# rgb = pd.read_csv(f'models/ensemble/Classifier_Prediction_Data/98/rgb/{name}--{fold}.csv')
# hsi = pd.read_csv(f'models/ensemble/Classifier_Prediction_Data/98/hsi/dense_net--{fold}.csv')
# from sklearn.metrics import classification_report
# import json
# y_true = rgb.iloc[:, -1]
# with open('Data/98/98_var_mappings.json', 'r') as file:
#     data = json.load(file)['id_to_class']
# target_names = []
# for i in range(98):
#   target_names.append(data[str(i)])

# _,_,y_pred = get_pred(rgb, hsi, 98 , w1)
# report = classification_report(y_true, y_pred, target_names=target_names)
# with open(f'models/ensemble/Classifier_Prediction_Data/{name}_ensemble_classification_report.txt', 'w') as file:
#   file.write(report)
  # print("Text saved successfully to", report)
# print(report)
