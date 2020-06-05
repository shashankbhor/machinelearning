#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:00:17 2019
Client for CNN
@author: base model by WildML portal
"""

import string
import nltk
import numpy as np
import os
import time
import datetime
import tensorflow as tf
from modelBuild import TextCNN
from nltk.corpus import stopwords
import pandas as pd
from tensorflow.contrib import learn
print(tf.__version__)

# Remove Punctuation from text
def remove_punctuation(text):
    tr_table = str.maketrans("", "", string.punctuation)
    return text.translate(tr_table)

def remove_stopwords(words_list):
    filtered_list = []
    for w in words_list:
        if w not in stop_words:
            filtered_list.append(w)
    return filtered_list

def stem_words(words_list):
    stemmed_words = [stemmer.stem(w.lower()) for w in words_list]
    return stemmed_words


def load_data_and_labels(df, categories):
    """
    Returns sentences and labels.
    """
    texts = []
    labels = []
    label_empty_list = [0] * len(categories)
    for index, row in df.iterrows():
        text = row["Cause_of_Injury"]
        category = row["category"]
        text = remove_punctuation(text)
        text = text.strip()
        # Generate labels
        label = list(label_empty_list)
        label[categories.index(category)] = 1
        texts.append(text)
        labels.append(label)
    return (texts, labels)


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
            
            
def preprocess():
    #Load data
    df = pd.read_csv("./data/CauseofInjurymodified.csv", encoding="latin-1")
    categories = list(df["category"].unique())
    # Load texts and labels
    texts, y = load_data_and_labels(df, categories)
    # Build vocabulary
    max_document_length = max([len(text.split(" ")) for text in texts])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(texts)))
    y = np.array(y)
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    test_sample_index = -1 * int(.1 * float(len(y)))
    x_train, x_test = x_shuffled[:test_sample_index], x_shuffled[test_sample_index:]
    y_train, y_test = y_shuffled[:test_sample_index], y_shuffled[test_sample_index:]
    del x, y, x_shuffled, y_shuffled
    return x_train, y_train, vocab_processor, x_test, y_test

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    with tf.Graph().as_default():
        session_conf=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess=tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(sequence_length=x_train.shape[1],
                          num_classes=y_train.shape[1],
                          vocab_size=len(vocab_processor.vocabulary_),
                          embedding_size=128,
                          filter_sizes=[3,4,5],
                          num_filters=128,
                          l2_reg_lambda=0.0)
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
            
            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 0.5
                }
                _, step, loss, accuracy = sess.run(
                    [train_op, global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                
                
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy = sess.run(
                    [global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                
                    
            # Generate batches
            batches = batch_iter(list(zip(x_train, y_train)), 64, 50)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 100 == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=None)
                    print("")
                if current_step % 100 == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))        


x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
train(x_train, y_train, vocab_processor, x_dev, y_dev)

