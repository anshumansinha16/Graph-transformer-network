import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from model_tf_3 import GTN 
import pdb
import pickle
import argparse
from utils import f1_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--file', '-f', type=str)
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--epoch', type=int, default=2,
                        help='Training Epochs')
    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2,
                        help='number of channels')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layer')
    parser.add_argument('--norm', type=str, default='true',
                        help='normalization')
    parser.add_argument('--adaptive_lr', type=str, default='false',
                        help='adaptive learning rate')

    print(parser.parse_args())
    
    args = parser.parse_args()
    
    print(args)
    
    epochs = args.epoch
    node_dim = args.node_dim
    num_channels = args.num_channels
    lr = args.lr
    weight_decay = args.weight_decay
    num_layers = args.num_layers
    norm = args.norm
    adaptive_lr = args.adaptive_lr

    with open('data/'+args.dataset+'/node_features.pkl','rb') as f:
        node_features = pickle.load(f)
    with open('data/'+args.dataset+'/edges.pkl','rb') as f:
        edges = pickle.load(f)
    with open('data/'+args.dataset+'/labels.pkl','rb') as f:
        labels = pickle.load(f)
        
    # creation of nodes, edges and labels in 3 separate matrix/ vectors!    
    
    num_nodes = edges[0].shape[0]
    num_nodes = 1000

    # A = Adjacency matrix 
    
    for i,edge in enumerate(edges): # i goesthrough numbers [0,1,2,3...] and edge through edges.
        if i ==0:
             A = tf.expand_dims(tf.convert_to_tensor(edge.todense(), dtype= tf.float32), -1)
        else:
             A = tf.concat((A,tf.expand_dims(tf.convert_to_tensor(edge.todense(), dtype= tf.float32), -1)), -1) 
    
    A = tf.concat((A, tf.expand_dims(tf.convert_to_tensor(tf.eye(num_nodes), dtype= tf.float32), -1) ), -1)
    
    node_features = node_features[0:1000,0:334]
    node_features = tf.convert_to_tensor(node_features, dtype= tf.float32)
    
    train_node = tf.convert_to_tensor(np.array(labels[0])[:,0])
    train_target = tf.convert_to_tensor(np.array(labels[0])[:,1])
    
    valid_node = tf.convert_to_tensor(np.array(labels[1])[:,0])
    valid_target = tf.convert_to_tensor(np.array(labels[1])[:,1])
    
    test_node = tf.convert_to_tensor(np.array(labels[2])[:,0])
    test_target = tf.convert_to_tensor(np.array(labels[2])[:,1])
    
    num_classes = tf.get_static_value(tf.reduce_max(train_target)) +1
    # num_classes = tf.math.maximum(train_target).item()+1
    
    final_f1 = 0
    
    for l in tf.range(1):
        
        model = GTN(num_edge=A.shape[-1],
                            num_channels=num_channels,
                            w_in = node_features.shape[1],
                            w_out = node_dim,
                            num_class=num_classes,
                            num_layers=num_layers,
                            norm=norm)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)

        # Train & Valid & Test
        best_val_loss = 10000
        best_test_loss = 10000
        best_train_loss = 10000
        best_train_f1 = 0
        best_val_f1 = 0
        best_test_f1 = 0
        

        for i in range(epochs):

            print("\nStart of epoch %d" % (i,))

            with tf.GradientTape() as tape:
              loss,y_train,Ws = model(A, node_features, train_node, train_target)
              train_f1 = tf.reduce_mean(f1_score(tf.math.argmax(y_train, 1), train_target, num_classes=num_classes)).cpu()
              print('Train - Loss: {}, Macro_F1: {}'.format(loss.cpu().numpy(), train_f1))

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if val_f1 > best_val_f1:
                best_val_loss = val_loss.cpu().numpy()
                best_train_loss = loss.cpu().numpy()
                best_train_f1 = train_f1
                best_val_f1 = val_f1

        print('---------------Best Results--------------------')
        print('Train - Loss: {}, Macro_F1: {}'.format(best_train_loss, best_train_f1))
        print('Valid - Loss: {}, Macro_F1: {}'.format(best_val_loss, best_val_f1))
        final_f1 += best_test_f1


# In[ ]:




