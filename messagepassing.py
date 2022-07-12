#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import inspect
import tensorflow as tf
from tensorflow import keras
import numpy as np

def tf_index_select(input_, dim, indices):
    """
    input_(tensor): input tensor
    dim(int): dimension
    indices(list): selected indices list
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape)-1
    shape[dim] = 1
    
    tmp = []
    for idx in indices:
        begin = [0]*len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)
    
    return res



special_args = [ 'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j']
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')


class MessagePassing(keras.layers.Layer):
    def __init__(self, aggr='add', flow='source_to_target'):
        super(MessagePassing, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.__message_args__ = inspect.getfullargspec(self.message)[0][1:]
        self.__special_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__message_args__)
                                 if arg in special_args]
        self.__message_args__ = [
            arg for arg in self.__message_args__ if arg not in special_args
        ]
        self.__update_args__ = inspect.getfullargspec(self.update)[0][2:]

    def propagate(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args = []
        for arg in self.__message_args__:
            if arg[-2:] in ij.keys():
                tmp = kwargs[arg[:-2]]
                if tmp is None:  # pragma: no cover
                    message_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if size[1 - idx] is None:
                            size[1 - idx] = tmp[1 - idx].size(0)
                        if size[1 - idx] != tmp[1 - idx].size(0):
                            raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    #tmp = torch.index_select(tmp, 0, edge_index[idx])
                    tmp = tf_index_select(tmp, 0, edge_index[idx])
                    message_args.append(tmp)
            else:
                message_args.append(kwargs[arg])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]

        out = self.message(*message_args)
        out = scatter_(self.aggr, out, edge_index[i], dim_size=size[i])
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        return x_j

    def update(self, aggr_out):  # pragma: no cover
        return 


# In[ ]:


tmp = torch.index_select(tmp, 0, edge_index[idx])


# In[1]:


def tf_index_select(input_, dim, indices):
    """
    input_(tensor): input tensor
    dim(int): dimension
    indices(list): selected indices list
    """
    shape = input_.get_shape().as_list()
    if dim == -1:
        dim = len(shape)-1
    shape[dim] = 1
    
    tmp = []
    for idx in indices:
        begin = [0]*len(shape)
        begin[dim] = idx
        tmp.append(tf.slice(input_, begin, shape))
    res = tf.concat(tmp, axis=dim)
    
    return res


# In[4]:


import tensorflow as tf
import numpy as np

a = np.arange(2*3*4).reshape(2,3,4)
dim = 1
indices = [0,2]

# pytorch


# In[5]:


res


# In[6]:


# tensorflow
res = tf_index_select(tf.constant(a), dim, indices)


# In[7]:


res


# In[9]:


res = torch.tensor(a).index_select(dim, torch.tensor(indices))


# In[ ]:




