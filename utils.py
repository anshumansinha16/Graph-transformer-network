#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np


def accuracy(pred, target):
    r"""Computes the accuracy of correct predictions.
    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
    :rtype: int
    """
    # return (pred == target).sum().item() / target.numel() 
    return tf.reduce_sum(tf.cast((pred == target), dtype=tf.int32)).numpy() / tf.size(target).numpy()


def true_positive(pred, target, num_classes):
    r"""Computes the number of true positive predictions.
    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.
    :rtype: :class:`LongTensor`
    """
    

    out = []
    
    for i in tf.range(num_classes):
        
        i = tf.cast(i, tf.int64)
        pred= tf.cast(pred, tf.int64)
        target= tf.cast(target, tf.int64)
        out.append(tf.math.count_nonzero((pred == i) & (target == i)))
        
    return tf.stack(out)


def true_negative(pred, target, num_classes):
    r"""Computes the number of true negative predictions.
    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.
    :rtype: :class:`LongTensor`
    """
    out = []
    for i in tf.range(num_classes):
        
        i = tf.cast(i, tf.int64)
        pred= tf.cast(pred, tf.int64)
        target= tf.cast(target, tf.int64)
        out.append(tf.math.count_nonzero((pred != i) & (target != i)))
        
        
    return tf.stack(out)



def false_positive(pred, target, num_classes):
    r"""Computes the number of false positive predictions.
    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.
    :rtype: :class:`LongTensor`
    """
    out = []
    for i in tf.range(num_classes):
        
        i = tf.cast(i, tf.int64)
        pred= tf.cast(pred, tf.int64)
        target= tf.cast(target, tf.int64)
        out.append(tf.math.count_nonzero((pred == i) & (target != i)))
        
        
    return tf.stack(out)



def false_negative(pred, target, num_classes):
    r"""Computes the number of false negative predictions.
    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.
    :rtype: :class:`LongTensor`
    """
    out = []
    for i in tf.range(num_classes):
        i = tf.cast(i, tf.int64)
        pred= tf.cast(pred, tf.int64)
        target= tf.cast(target, tf.int64)
        out.append(tf.math.count_nonzero((pred != i) & (target == i)))
    return tf.stack(out)



def precision(pred, target, num_classes):
    r"""Computes the precision:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}`.
    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.
    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes)
    fp = false_positive(pred, target, num_classes)

    out = tp / (tp + fp)
    out = tf.where(tf.math.is_nan(out), 0., out)

    return out



def recall(pred, target, num_classes):
    r"""Computes the recall:
    :math:`\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}`.
    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.
    :rtype: :class:`Tensor`
    """
    tp = true_positive(pred, target, num_classes)
    fn = false_negative(pred, target, num_classes)

    out = tp / (tp + fn)
    out = tf.where(tf.math.is_nan(out), 0., out)

    return out



def f1_score(pred, target, num_classes):
    r"""Computes the :math:`F_1` score:
    :math:`2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}
    {\mathrm{precision}+\mathrm{recall}}`.
    Args:
        pred (Tensor): The predictions.
        target (Tensor): The targets.
        num_classes (int): The number of classes.
    :rtype: :class:`Tensor`
    """
    prec = precision(pred, target, num_classes)
    rec = recall(pred, target, num_classes)

    score = 2 * (prec * rec) / (prec + rec)
    score = tf.where(tf.math.is_nan(score), 0., score)

    return score


