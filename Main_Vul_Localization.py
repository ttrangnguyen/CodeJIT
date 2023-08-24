import argparse

from vul_localization.STM_blstm import *
from vul_localization.Result_Manager import *
from gensim.models import KeyedVectors
import pandas





def read_train_and_test_data():
    tmp_file = open("Data/JIT_DATASET/data_split/suffle/train_cross_id.txt", "r").readlines()
    train_ids = [f.replace("\n", "") for f in tmp_file]
    data = pandas.read_csv("Data/vul_stmt.csv")
    train_data = data[data["commit_id"].isin(train_ids)]
    train_data.to_csv("Data/train_examples.csv")
    train_data.to_csv("Data/test_examples.csv")


def main(_vector_length, _max_seq_length, _max_code_stmt_length, word2vec_model_path,
         training_data_file, testing_data_file, mode_name, batch_size, epochs, output_file):
    vector_length = _vector_length
    max_seq_length = _max_seq_length
    max_code_stmt_length = _max_code_stmt_length
    vectorizer = KeyedVectors.load(word2vec_model_path, mmap='r')
    blstm = STM_Train_BLSTM(vectorizer, data_file=training_data_file,
                            name=mode_name, vector_length = vector_length,
                            max_seq_length=max_seq_length, max_code_stmt_length=max_code_stmt_length, batch_size = batch_size)

    blstm.train(epochs=epochs)
    blstm = STM_Test_BLSTM(vectorizer, data_file=testing_data_file,
                           name=mode_name,vector_length = vector_length,
                           max_seq_length=max_seq_length, max_code_stmt_length=max_code_stmt_length, batch_size = batch_size)
    predictions, targets = blstm.test()

    classification_accuracy_report(predictions, targets, 0.5)
    predicted_labels = []
    for predicts in predictions:
        for item in predicts:
            predicted_labels.append(item)
    expected_labels = []
    for tar in targets:
        for item in tar:
            expected_labels.append(item)

    data = pandas.read_csv(testing_data_file, encoding='utf-8')

    data["predicted_labels"] = predicted_labels
    data["expected_labels"] = expected_labels
    data.to_csv(output_file, encoding='utf-8')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_length', type=int, default=32)
    parser.add_argument('--max_seq_length', type=int, default=600)
    parser.add_argument('--max_code_stmt_length', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--w2vmodel_path', type=str, default="Model/word2vec.wordvectors")
    parser.add_argument('--training_data_file', type=str)
    parser.add_argument('--testing_data_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--model_name', type=str, default="vul_localization")
    args = parser.parse_args()

    vector_length = args.vector_length
    max_seq_length = args.max_seq_length
    max_code_stmt_length = args.max_code_stmt_length
    batch_size = args.batch_size
    epochs = args.epochs
    model_name = args.model_name
    w2vmodel_path = args.w2vmodel_path
    training_data_file = args.training_data_file
    testing_data_file = args.testing_data_file
    output_file = args.output_file
    main(vector_length, max_seq_length, max_code_stmt_length, w2vmodel_path, training_data_file,
         testing_data_file,model_name,batch_size, epochs, output_file)
    #read_train_and_test_data()
