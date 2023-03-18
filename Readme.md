##Instruction to train word2vec model
```
python Main_Word2Vec.py --vocab_file_path="vocab/word2vec.vocabulary.txt" --vector_length=32 --model_dir="model" --model_name="w2v"
```
The vocab file can be downloaded <a href="https://"> here </a>

Or you can used our <a href="https://"> pre-trained model </a>

##Instruction to embed features of nodes and edges of the graphs
```
python Main_Graph_Embedding.py --node_graph_dir="Data/Graph/node" --node_graph_dir="Data/Graph/edge" --label=1 --embedding_graph_dir="Data/embedding" 
```

Or you can used our <a href="https://"> embedded graphs </a>

##Instruction to train and test GNN models

```
!python 'Main_VULJIT_Detection.py' --graph_dir='Data/embedding'  --train_file='Data/data_split/suffle/train_time_id.txt' --test_file='Data/data_split/suffle/test_time_id.txt'  --model_dir='Model'  --model_name="rgcn" 
```
