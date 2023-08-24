import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def ranking_precision_and_recall_score(y_true, y_score, k=10):

    if k < 1:
      k = round(len(y_true) * k)
    
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = unique_y[1]
    n_pos = np.sum(y_true == pos_label)
    
    order = np.lexsort((1 - y_true,y_score))[::-1]

    y_true = np.take(y_true, order[:k])
    
    n_relevant = np.sum(y_true == pos_label)
    
    try:    
        precision = float(n_relevant) / min(len(y_true), k)
        recall = float(n_relevant) / n_pos
        f1_score = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        print("ZeroDivisionError")
        precision = 0.0
        recall = 0.0
        f1_score = 0.0

    return precision, recall, f1_score

def print_report(precision, recall, f1_score, k, length_seq):
    print("\t\t\t\tPrecison\t\t\tRecall\t\t\tF1_scrore")
    print()
    print(f"Ranking top@{k * 100}% ({int(k * length_seq)})\t\t{round(precision, 3)}\t\t\t\t{round(recall, 3)}\t\t\t{round(f1_score, 3)}")

def stm_level_test(blstm, data, thres_hold = 0.5):
    predictions = blstm.model.predict(blstm.X_test)[:,1]
    y_test = blstm.y_test[:,1]
    lstm_predicted_labels = predictions.round()

    ks = [0.01, 0.05, 0.1, 0.2, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    result = {
        "LSTM": {}, 
        # "RF_layer_1": {},
        # "RF_layer_2": {},
        # "RF_layer_5": {}
    }

    print("LSTM")
    for k in ks:
        precision, recall, f1_score = ranking_precision_and_recall_score(y_test, predictions, k)
        result["LSTM"][str(k)] = ( precision, recall)
        print_report( precision, recall, f1_score, k, len(y_test))

    print(classification_report(y_test, lstm_predicted_labels))
    predicted_results = []
    for item in lstm_predicted_labels:
      predicted_results.append(item.astype(int))

    result_file = f"results/classification_result.txt"
    # with open(result_file, "w") as output:
    #   for i in range(0, len(test_index)):
    #     output.write("expected:" + str(data.at[test_index[i], TARGET]) + "\n")
    #     output.write("predicted:" + str(predicted_results[i]) + "\n")
    #     output.write("code_stmt: " + data.at[test_index[i], CODE_STMT].replace(" ","") + "\n" )
    #     # output.write("code_line: " + data.at[test_index[i], CODE_LINK] + "\n")
    #     # output.write("context: " + data.at[test_index[i], CTX_SURROUNDING] + "\n")
    #     # output.write("flaw_line: " + data.at[test_index[i], FLAW_LINE] + "\n")
    #     output.write("----------------------------------------\n")
    return result

def classification_accuracy_report(predictions, targets, r):
  predicted_lables = []
  for predicts in predictions:
      for item in predicts:
        if item[1] >= r:
          predicted_lables.append(1)
        else:
          predicted_lables.append(0)
  expected_lables = []
  for tar in targets:
    for item in tar:
      expected_lables.append(item)
  print(classification_report(expected_lables, predicted_lables))
