# -*- coding: utf-8 -*-
"""LSTM Encoder
   Copyright 2018 Mitsubishi Electric Research Labs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio


class LSTMEncoder(nn.Module):

    def __init__(self, n_layers, in_size, out_size, embed_size, dropout=0.5, initialEmbW=None):
        """Initialize encoder with structure parameters
        Args:
            n_layers (int): Number of layers.
            in_size (int): Dimensionality of input vectors.
            out_size (int) : Dimensionality of hidden vectors to be output.
            embed_size (int): Dimensionality of word embedding.
            dropout (float): Dropout ratio.
        """
        super(LSTMEncoder, self).__init__()
        self.embed =nn.Embedding(in_size, embed_size)
        self.lstm = nn.LSTM(embed_size,out_size,n_layers,dropout,batch_first=True)



    def __call__(self, s, xs, **kwargs):
        """Calculate all hidden states and cell states.
        Args:
            s  (~chainer.Variable or None): Initial (hidden & cell) states. If ``None``
                is specified zero-vector is used.
            xs (list of ~chianer.Variable): List of input sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.
        Return:
            (hy,cy): a pair of hidden and cell states at the end of the sequence,
            ys: a hidden state sequence at the last layer
        """
        if len(xs) != 0:
            sections = np.array([len(x) for x in xs], dtype=np.int32)
            # aa = self.embed(torch.tensor(xs[0][0],dtype=torch.long).cuda())
            aa = torch.cat(xs, 0)
            bb = self.embed(torch.tensor(aa, dtype=torch.long).cuda())
            cc = sections.tolist()
            wj = torch.split(bb, cc,dim=0)
            wj = list(wj)
            #sorting
            sort_wj=[]
            cc = torch.from_numpy(sections)
            cc, perm_index = torch.sort(cc, 0, descending=True)
            sort_wj.append([wj[i] for i in perm_index])
            padded_wj = nn.utils.rnn.pad_sequence(sort_wj[0], batch_first=True)
            packed_wj = nn.utils.rnn.pack_padded_sequence(padded_wj, list(cc.data), batch_first=True)
            # ys,(hy,cy) = self.lstm(packed_wj)
            # ys = nn.utils.rnn.pad_packed_sequence(ys, batch_first=True)[0]


            # restore the sorting
            # odx = perm_index.view(-1, 1).unsqueeze(1).expand(ys.size(0), ys.size(1), ys.size(2))
            # ys2 = ys.gather(0, odx.cuda())

            # idx = (cc - 1).view(-1, 1).expand(ys.size(0), ys.size(2)).unsqueeze(1)
            # idx = torch.tensor(idx, dtype=torch.long)
            # decoded = ys.gather(1, idx.cuda()).squeeze()
            #
            # # restore the sorting
            # odx = perm_index.view(-1, 1).expand(ys.size(0), ys.size(-1))
            # decoded = decoded.gather(0, odx.cuda())


            # a1=ys2.cpu()
            # a1=a1.data.numpy()
            # a2=ys.cpu()
            # a2=a2.data.numpy()
            # a3=perm_index.cpu()
            # a3=a3.data.numpy()
            # a4=decoded.cpu()
            # a4=a4.data.numpy()
            # sio.savemat('np_vector2.mat', {'a1':a1,'a2':a2,'a3':a3,'a4':a4})
        else:
            hx = [ self.embed(xs[0]) ]
        if s is not None:
            ys, (hy,cy) = self.lstm(packed_wj,(s[0], s[1]))
        else:
            ys, (hy,cy) = self.lstm(packed_wj)
        #resorting
        ys = nn.utils.rnn.pad_packed_sequence(ys, batch_first=True)[0]
        if len(xs)>1:
            idx = (cc - 1).view(-1, 1).expand(ys.size(0), ys.size(2)).unsqueeze(1)
            idx = torch.tensor(idx, dtype=torch.long)
            decoded = ys.gather(1, idx.cuda()).squeeze()

            # restore the sorting
            cc2, perm_index2 = torch.sort(perm_index, 0)
            odx = perm_index2.view(-1, 1).expand(ys.size(0), ys.size(-1))
            decoded = decoded.gather(0, odx.cuda())
        else:
            decoded = ys[:,-1,:]

        return decoded

