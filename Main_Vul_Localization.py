import argparse
from jitvul_model.vul_localization import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_graph_dir', type=str, help='dir of graph data')
    parser.add_argument('--test_graph_dir', type=str, help='file of training data')
    parser.add_argument('--model_dir', type=str, help='output of trained model',
                        default='Model')
    parser.add_argument('--model_name', type=str, help='name of the model',
                        default='vul_localization')
    parser.add_argument('--GNN_type', default= "RGCN")
    parser.add_argument('--graph_readout_func', default= "add")
    parser.add_argument('--mode', default= "train_and_test")
    parser.add_argument('--hidden_size', default=  32)
    parser.add_argument('--learning_rate', default=  0.0001)
    parser.add_argument('--dropout_rate', default= 0.2)
    parser.add_argument('--max_epochs', default= 50)
    parser.add_argument('--num_of_layers', default= 2)

    args = parser.parse_args()

    train_graph_dir = args.train_graph_dir
    test_graph_dir = args.test_graph_dir
    model_path = args.model_dir
    mode = args.mode
    params = {'max_epochs': int(args.max_epochs), 'hidden_size': int(args.hidden_size), 'lr': float(args.learning_rate), 'dropout_rate': float(args.dropout_rate),
              "num_of_layers": int(args.num_of_layers), 'model_name': args.model_name, 'GNN_type': args.GNN_type, "graph_readout_func": args.graph_readout_func}
    if mode == "train_and_test":
        print()
        print("Training............")
        print()
        train_model(train_file_path = train_graph_dir, test_file_path = test_graph_dir, _params=params, model_path = model_path)
        print()
        print("Testing..............")
        print()
        test_model(test_file_path = test_graph_dir, _params=params, model_path = model_path)
    elif mode == "train_only":
        print()
        print("Training............")
        print()
        train_model(train_file_path = train_graph_dir, test_file_path = test_graph_dir, _params=params, model_path = model_path)
    elif mode == "test_only":
        print()
        print("Testing..............")
        print()
        test_model(test_file_path = train_graph_dir, _params=params, model_path = model_path)
    else:
        print("Mode " + mode + " is not supported.")
