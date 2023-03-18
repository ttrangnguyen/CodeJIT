import argparse

from Word2Vec.Word2Vec import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file_path', type=str, help='path of the vocabulary file')
    parser.add_argument('--vector_length', type=int, help='length of the embedding vector', default=32)
    parser.add_argument('--model_dir', type=str, help='dir to save model', default="model")
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    vocabulary_file_path = args.vocab_file_path
    vector_length = int(args.vector_length)
    model_name = args.model_name
    model_dir = args.model_dir
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    vocab_f = open(vocabulary_file_path, "r")
    sentences = [x.replace("\n", "").split(" ") for x in vocab_f.readlines()]

    print("sentence size:", len(sentences))
    vectorizer = Word2VecTrain(sentences, vector_length, os.path.join(model_dir, model_name))
    vectorizer.train_model()
