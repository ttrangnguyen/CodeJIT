import argparse
from jitvul_model.jit_vul_detection_model import  *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', type=str, help='dir of graph data')
    parser.add_argument('--train_file', type=str, help='file of training data')
    parser.add_argument('--test_file', type=str, help='file of testing data')
    parser.add_argument('--model_dir', type=str, help='output of trained model',
                        default='Model')
    parser.add_argument('--model_name', type=str, help='name of the model',
                        default='best_model')
    parser.add_argument('--GNN_type', default= "RGCN")
    parser.add_argument('--graph_readout_func', default= "add")
    parser.add_argument('--mode', default=  "train_and_test")
    parser.add_argument('--hidden_size', default=  32)
    parser.add_argument('--learning_rate', default=  0.0001)
    parser.add_argument('--dropout_rate', default= 0.2)
    parser.add_argument('--max_epochs', default= 50)
    parser.add_argument('--num_of_layers', default= 2)

    args = parser.parse_args()

    graph_path = args.graph_dir
    train_path = args.train_file
    test_path = args.test_file
    model_path = args.model_dir
    mode = args.mode
    params = {'max_epochs': int(args.max_epochs), 'hidden_size': int(args.hidden_size), 'lr': float(args.learning_rate), 'dropout_rate': float(args.dropout_rate),
              "num_of_layers": int(args.num_of_layers), 'model_name': args.model_name, 'GNN_type': args.GNN_type, "graph_readout_func": args.graph_readout_func}
    if mode == "train_and_test":
        print()
        print("Training............")
        print()
        train_model(graph_path = graph_path, train_file_path = train_path, test_file_path = test_path, _params=params, model_path = model_path)
        print()
        print("Testing..............")
        print()
        test_model(graph_path = graph_path, test_file_path = test_path, _params=params, model_path = model_path)
    elif mode == "train_only":
        print()
        print("Training............")
        print()
        train_model(graph_path = graph_path, train_file_path = train_path, test_file_path = test_path, _params=params, model_path = model_path)
    elif mode == "test_only":
        print()
        print("Testing..............")
        print()
        test_model(graph_path = graph_path, test_file_path = test_path, _params=params, model_path = model_path)
    else:
        print("Mode " + mode + " is not supported.")
