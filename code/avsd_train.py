#!/usr/bin/env python
"""Train an encoder-decoder model for audio-visual scene-aware dialog
   Copyright 2018 Mitsubishi Electric Research Labs
"""

import argparse
import logging
import math
import sys
import time
import random
import os
import json

import numpy as np
import pickle
import six
import threading

import torch
import torch.nn as nn

import data_handler as dh

from mmseq2seq_model import MMSeq2SeqModel
from multimodal_encoder import MMEncoder
from lstm_encoder import LSTMEncoder
from hlstm_encoder import HLSTMEncoder
from hlstm_decoder import HLSTMDecoder


def fetch_batch(dh, data, index, result):
    result.append(dh.make_batch(data, index))

# Evaluation routine
def evaluate(model, data, indices):
    start_time = time.time()
    eval_loss = 0.
    eval_num_words = 0
    model.eval()
    with torch.no_grad():
        # fetch the first batch
        batch = [dh.make_batch(data, indices[0])]
        # evaluation loop
        for j in six.moves.range(len(indices)):
            # get a fetched batch
            x_batch, h_batch, q_batch, a_batch_in, a_batch_out = batch.pop()
            # fetch the next batch in parallel
            if j < len(indices)-1:
                prefetch = threading.Thread(target=fetch_batch, 
                                args=([dh, data, indices[j+1], batch]))
                prefetch.start()
            # propagate for training
            x = [torch.from_numpy(x) for x in x_batch]
            h = [[torch.from_numpy(h) for h in hb] for hb in h_batch]
            q = [torch.from_numpy(q) for q in q_batch]
            ai = [torch.from_numpy(ai) for ai in a_batch_in]
            ao = [torch.from_numpy(ao) for ao in a_batch_out]
            _, _, loss = model.loss(x, h, q, ai, ao)

            num_words = sum([len(s) for s in ao])
            eval_loss += loss.cpu().data.numpy() * num_words
            eval_num_words += num_words
            # wait prefetch completion
            prefetch.join()
    model.train()

    wall_time = time.time() - start_time
    return math.exp(eval_loss/eval_num_words), wall_time


##################################
# main
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    # train, dev and test data
    parser.add_argument('--vocabfile', default='', type=str, 
                        help='Vocabulary file (.json)')
    parser.add_argument('--dictmap', default='', type=str, 
                        help='Dict id-map file (.json)')
    parser.add_argument('--fea-type', nargs='+', type=str, 
                        help='Image feature files (.pkl)')
    parser.add_argument('--train-path', default='', type=str,
                        help='Path to training feature files')
    parser.add_argument('--train-set', default='', type=str,
                        help='Filename of train data')
    parser.add_argument('--valid-path', default='', type=str,
                        help='Path to validation feature files')
    parser.add_argument('--valid-set', default='', type=str,
                        help='Filename of validation data')
    parser.add_argument('--include-caption', action='store_true',
                        help='Include caption in the history')
    # Attention model related
    parser.add_argument('--model', '-m', default='', type=str,
                        help='Attention model to be output')
    parser.add_argument('--num-epochs', '-e', default=15, type=int,
                        help='Number of epochs')
    # multimodal encoder parameters
    parser.add_argument('--enc-psize', '-p', nargs='+', type=int,
                        help='Number of projection layer units')
    parser.add_argument('--enc-hsize', '-u', nargs='+', type=int,
                        help='Number of hidden units')
    parser.add_argument('--att-size', '-a', default=100, type=int,
                        help='Number of attention layer units')
    parser.add_argument('--mout-size', default=100, type=int,
                        help='Number of output layer units')
    # input (question) encoder parameters
    parser.add_argument('--embed-size', default=200, type=int, 
                        help='Word embedding size')
    parser.add_argument('--in-enc-layers', default=2, type=int,
                        help='Number of input encoder layers')
    parser.add_argument('--in-enc-hsize', default=200, type=int,
                        help='Number of input encoder hidden layer units')
    # history (QA pairs) encoder parameters
    parser.add_argument('--hist-enc-layers', nargs='+', type=int,
                        help='Number of history encoder layers')
    parser.add_argument('--hist-enc-hsize', default=200, type=int,
                        help='History embedding size')
    parser.add_argument('--hist-out-size', default=200, type=int,
                        help='History embedding size')
    # response (answer) decoder parameters
    parser.add_argument('--dec-layers', default=2, type=int,
                        help='Number of decoder layers')
    parser.add_argument('--dec-psize', '-P', default=200, type=int,
                        help='Number of decoder projection layer units')
    parser.add_argument('--dec-hsize', '-d', default=200, type=int,
                        help='Number of decoder hidden layer units')
    # Training conditions
    parser.add_argument('--optimizer', '-o', default='AdaDelta', type=str,
                        choices=['SGD', 'Adam', 'AdaDelta', 'RMSprop'],
                        help="optimizer")
    parser.add_argument('--rand-seed', '-s', default=1, type=int, 
                        help="seed for generating random numbers")
    parser.add_argument('--batch-size', '-b', default=20, type=int,
                        help='Batch size in training')
    parser.add_argument('--max-length', default=20, type=int,
                        help='Maximum length for controling batch size')
    # others
    parser.add_argument('--verbose', '-v', default=0, type=int,
                        help='verbose level')

    args = parser.parse_args()

    random.seed(args.rand_seed)
    np.random.seed(args.rand_seed)

    if args.dictmap != '':
        dictmap = json.load(open(args.dictmap, 'r'))
    else:
        dictmap = None

    if args.verbose >= 1:
        logging.basicConfig(level=logging.DEBUG, 
            format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s %(levelname)s: %(message)s')

    logging.info('Command line: ' + ' '.join(sys.argv))
    # get vocabulary
    logging.info('Extracting words from ' + args.train_set)
    vocab = dh.get_vocabulary(args.train_set, include_caption=args.include_caption)
    # load data
    logging.info('Loading training data from ' + args.train_set)
    train_data = dh.load(args.fea_type, args.train_path, args.train_set, 
                         vocabfile=args.vocabfile, 
                         include_caption=args.include_caption, 
                         vocab=vocab, dictmap=dictmap)

    logging.info('Loading validation data from ' + args.valid_set)
    valid_data = dh.load(args.fea_type, args.valid_path, args.valid_set, 
                         vocabfile=args.vocabfile, 
                         include_caption=args.include_caption, 
                         vocab=vocab, dictmap=dictmap)

    feature_dims = dh.feature_shape(train_data)
    logging.info("Detected feature dims: {}".format(feature_dims));

    # Prepare RNN model and load data
    model = MMSeq2SeqModel(
                MMEncoder(feature_dims, args.mout_size, enc_psize=args.enc_psize,
                          enc_hsize=args.enc_hsize, att_size=args.att_size,
                          state_size=args.in_enc_hsize),
                HLSTMEncoder(args.hist_enc_layers[0], args.hist_enc_layers[1],
                          len(vocab), args.hist_out_size, args.embed_size,
                          args.hist_enc_hsize),
                LSTMEncoder(args.in_enc_layers, len(vocab), args.in_enc_hsize,
                          args.embed_size),
                HLSTMDecoder(args.dec_layers, len(vocab), len(vocab), args.embed_size,
                          args.mout_size + args.hist_out_size + args.in_enc_hsize,
                          args.dec_hsize, args.dec_psize,
                          independent=True))

    # report data summary
    logging.info('#vocab = %d' % len(vocab))
    # make batchset for training
    logging.info('Making mini batches for training data')
    train_indices, train_samples = dh.make_batch_indices(train_data, args.batch_size,
                                                         max_length=args.max_length)
    logging.info('#train sample = %d' % train_samples)
    logging.info('#train batch = %d' % len(train_indices))
    # make batchset for validation
    logging.info('Making mini batches for validation data')
    valid_indices, valid_samples = dh.make_batch_indices(valid_data, args.batch_size,
                                                     max_length=args.max_length)
    logging.info('#validation sample = %d' % valid_samples)
    logging.info('#validation batch = %d' % len(valid_indices))
    # copy model to gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # save meta parameters
    path = args.model + '.conf'
    with open(path, 'wb') as f:
        pickle.dump((vocab, args), f, -1)
 
    # start training 
    logging.info('----------------')
    logging.info('Start training')
    logging.info('----------------')
    # Setup optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters())
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == 'AdaDelta':
        optimizer = torch.optim.Adadelta(model.parameters())
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters())
   
    # initialize status parameters
    modelext = '.pth.tar'
    cur_loss = 0.
    cur_num_words = 0
    epoch = 0
    start_at = time.time()
    cur_at = start_at
    min_valid_ppl = 1.0e+10
    n = 0
    report_interval = 1000/args.batch_size
    bestmodel_num = 0
    random.shuffle(train_indices)
    # do training iterations
    for i in six.moves.range(args.num_epochs):
        logging.info('Epoch %d : %s' % (i+1, args.optimizer))
        train_loss = 0.
        train_num_words = 0
        # fetch the first batch
        batch = [dh.make_batch(train_data, train_indices[0])]
        # train iterations
        for j in six.moves.range(len(train_indices)):
            # get fetched batch
            x_batch, h_batch, q_batch, a_batch_in, a_batch_out = batch.pop()
            # fetch the next batch in parallel
            if j < len(train_indices)-1:
                prefetch = threading.Thread(target=fetch_batch, 
                                args=([dh, train_data, train_indices[j+1], batch]))
                prefetch.start()
            
            # propagate for training
            x = [torch.from_numpy(x) for x in x_batch]
            h = [[torch.from_numpy(h) for h in hb] for hb in h_batch]
            q = [torch.from_numpy(q) for q in q_batch]
            ai = [torch.from_numpy(ai) for ai in a_batch_in]
            ao = [torch.from_numpy(ao) for ao in a_batch_out]
            _, _, loss = model.loss(x, h, q, ai, ao)

            num_words = sum([len(s) for s in ao])
            batch_loss = loss.cpu().data.numpy()
            train_loss += batch_loss * num_words
            train_num_words += num_words

            cur_loss += batch_loss * num_words
            cur_num_words += num_words
            if (n + 1) % report_interval == 0:
                now = time.time()
                throuput = report_interval / (now - cur_at)
                perp = math.exp(cur_loss / cur_num_words)
                logging.info('iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(n + 1, perp, throuput))
                cur_at = now
                cur_loss = 0.
                cur_num_words = 0
            n += 1
            # Run truncated BPTT
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # wait prefetch completion
            prefetch.join()

        logging.info("epoch: %d  train perplexity: %f" % (i+1, math.exp(train_loss/train_num_words)))

        # validation step 
        logging.info('-----------------------validation--------------------------')
        now = time.time()
        valid_ppl, valid_time = evaluate(model, valid_data, valid_indices)
        logging.info('validation perplexity: %.4f' % (valid_ppl))
        
        # update the model via comparing with the lowest perplexity  
        modelfile = args.model + '_' + str(i + 1) + modelext
        logging.info('writing model params to ' + modelfile)
        torch.save(model, modelfile)

        if min_valid_ppl > valid_ppl:
            bestmodel_num = i+1
            logging.info('validation perplexity reduced %.4f -> %.4f' % (min_valid_ppl, valid_ppl))
            min_valid_ppl = valid_ppl

        cur_at += time.time() - now  # skip time of evaluation and file I/O
        logging.info('----------------')

    # make a symlink to the best model
    logging.info('the best model is epoch %d.' % bestmodel_num)
    logging.info('a symbolic link is made as ' + args.model + '_best' + modelext)
    if os.path.exists(args.model + '_best' + modelext):
        os.remove(args.model + '_best' + modelext)
    os.symlink(os.path.basename(args.model + '_' + str(bestmodel_num) + modelext),
               args.model + '_best' + modelext)
    logging.info('done')
