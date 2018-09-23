# -*- coding: utf-8 -*-
"""Hierarchical LSTM Encoder
   Copyright 2018 Mitsubishi Electric Research Labs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import six
import scipy.io as sio

class HLSTMEncoder(nn.Module):

    def __init__(self, n_wlayers, n_slayers, in_size, out_size, embed_size, hidden_size, dropout=0.5, ignore_label=None, initialEmbW=None, independent=False):
        """Initialize encoder with structure parameters
        Args:
            n_layers (int): Number of layers.
            in_size (int): Dimensionality of input vectors.
            out_size (int) : Dimensionality of hidden vectors to be output.
            embed_size (int): Dimensionality of word embedding.
            dropout (float): Dropout ratio.
        """
        super(HLSTMEncoder, self).__init__()
        self.embed = nn.Embedding(in_size, embed_size)
        self.wlstm = nn.LSTM(embed_size,hidden_size,n_wlayers,dropout,batch_first=True)
        self.slstm = nn.LSTM(hidden_size,out_size,n_slayers,dropout,batch_first=True)

        self.independent = independent


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
        # word level within sentence
        sx = []
        for l in six.moves.range(len(xs)):
            if len(xs[l]) != 0:
                sections = np.array([len(x) for x in xs[l]], dtype=np.int32)
                aa = torch.cat(xs[l], 0)
                bb = self.embed(torch.tensor(aa, dtype=torch.long).cuda())
                cc = sections.tolist()
                wj = torch.split(bb, cc, dim=0)
                wj = list(wj)
                # sorting
                sort_wj = []
                cc = torch.from_numpy(sections)
                cc, perm_index = torch.sort(cc, 0, descending=True)
                sort_wj.append([wj[i] for i in perm_index])
                padded_wj = nn.utils.rnn.pad_sequence(sort_wj[0], batch_first=True)
                packed_wj = nn.utils.rnn.pack_padded_sequence(padded_wj, list(cc.data), batch_first=True)
            else:
                xl = [ self.embed(xs[l][0]) ]
            if hasattr(self, 'independent') and self.independent:
                ys, (why, wcy) = self.wlstm(packed_wj)
            else:
                if l==0:
                    ys, (why, wcy) = self.wlstm(packed_wj)
                else:
                    ys, (why, wcy) = self.wlstm(packed_wj, (why, wcy))
            ys = nn.utils.rnn.pad_packed_sequence(ys, batch_first=True)[0]
            if len(xs[l]) > 1:
                idx = (cc - 1).view(-1, 1).expand(ys.size(0), ys.size(2)).unsqueeze(1)
                idx = torch.tensor(idx, dtype=torch.long)
                decoded = ys.gather(1, idx.cuda()).squeeze()

                # restore the sorting
                cc2, perm_index2 = torch.sort(perm_index, 0)
                odx = perm_index2.view(-1, 1).expand(ys.size(0), ys.size(-1))
                decoded = decoded.gather(0, odx.cuda())
            else:
                decoded = ys[:, -1, :]

            sx.append(decoded)

        # sentence level
        sxs = torch.stack(sx, dim=0)
        sxs = sxs.permute(1,0,2)
        # sxl = [sxs[i] for i in six.moves.range(len(sxs))]
        if s is not None:
            sys, (shy, scy) = self.slstm( sxs, (s[0], s[1]))
        else:
            sys, (shy, scy) = self.slstm( sxs )

        return shy

