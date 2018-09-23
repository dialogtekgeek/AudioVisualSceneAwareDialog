# -*- coding: utf-8 -*-
"""Hierarchical LSTM Decoder module
   Copyright 2018 Mitsubishi Electric Research Labs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import six

class HLSTMDecoder(nn.Module):

    frame_based = False
    take_all_states = False

    def __init__(self, n_layers, in_size, out_size, embed_size, in_size_hier, hidden_size, proj_size, dropout=0.5, initialEmbW=None, independent=False):
        """Initialize encoder with structure parameters

        Args:
            n_layers (int): Number of layers.
            in_size (int): Dimensionality of input vectors.
            out_size (int): Dimensionality of output vectors.
            embed_size (int): Dimensionality of word embedding.
            hidden_size (int) : Dimensionality of hidden vectors.
            proj_size (int) : Dimensionality of projection before softmax.
            dropout (float): Dropout ratio.
        """
        super(HLSTMDecoder, self).__init__()
        self.embed = nn.Embedding(in_size, embed_size)
        self.lstm = nn.LSTM(embed_size+in_size_hier,hidden_size,n_layers,dropout,batch_first=True)
        self.proj = nn.Linear(hidden_size, proj_size)
        self.out = nn.Linear(proj_size, out_size)

        self.n_layers = n_layers
        self.dropout = dropout
        self.independent = independent


    def __call__(self, s, hs, xs):
        """Calculate all hidden states, cell states, and output prediction.

        Args:
            s (~chainer.Variable or None): Initial (hidden, cell) states.  If ``None``
                is specified zero-vector is used.
            hs (list of ~chianer.Variable): List of input state sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.
            xs (list of ~chianer.Variable): List of input label sequences.
                Each element ``xs[i]`` is a :class:`chainer.Variable` holding
                a sequence.
        Return:
            (hy,cy): a pair of hidden and cell states at the end of the sequence,
            y: a sequence of pre-activatin vectors at the output layer
 
        """
        if len(xs) > 1:
            sections = np.array([len(x) for x in xs], dtype=np.int32)
            aa = torch.cat(xs, 0)
            bb = self.embed(torch.tensor(aa, dtype=torch.long).cuda())
            cc = sections.tolist()
            hx = torch.split(bb, cc, dim=0)
        else:
            hx = [ self.embed(xs[0]) ]

        hxc = [ torch.cat((hx[i], hs[i].repeat(hx[i].shape[0], 1)), dim=1) for i in six.moves.range(len(hx))]

        sort_hxc = []
        cc = torch.from_numpy(sections)
        cc, perm_index = torch.sort(cc, 0, descending=True)
        sort_hxc.append([hxc[i] for i in perm_index])
        padded_hxc = nn.utils.rnn.pad_sequence(sort_hxc[0], batch_first=True)
        packed_hxc = nn.utils.rnn.pack_padded_sequence(padded_hxc, list(cc.data), batch_first=True)
        if s is None or (hasattr(self, 'independent') and self.independent):
            ys, (hy, cy) = self.lstm(packed_hxc)
            ys = nn.utils.rnn.pad_packed_sequence(ys, batch_first=True)[0]

            # restore the sorting
            cc2, perm_index2 = torch.sort(perm_index, 0)
            odx = perm_index2.view(-1, 1).unsqueeze(1).expand(ys.size(0), ys.size(1), ys.size(2))
            ys2 = ys.gather(0, odx.cuda())

            ys2_list=[]
            ys2_list.append([ys2[i,0:sections[i],:] for i in six.moves.range(ys2.shape[0])])
        y = self.out(self.proj(
                F.dropout(torch.cat(ys2_list[0],dim=0), p=self.dropout)))
        return (hy,cy),y


    # interface for beam search
    def initialize(self, s, x, i):
        """Initialize decoder

        Args:
            s (any): Initial (hidden, cell) states.  If ``None`` is specified
                     zero-vector is used.
            x (~chainer.Variable or None): Input sequence
            i (int): input label.
        Return:
            initial decoder state
        """
        # LSTM decoder can be initialized in the same way as update()
        if len(x) > 1:
            self.hx = F.vstack([x[j][-1] for j in six.moves.range(len(x[1]))])
        else:
            self.hx = x
        if hasattr(self, 'independent') and self.independent:
            return self.update(None,i)
        else:
            return self.update(s,i)


    def update(self, s, i):
        """Update decoder state

        Args:
            s (any): Current (hidden, cell) states.  If ``None`` is specified 
                     zero-vector is used.
            i (int): input label.
        Return:
            (~chainer.Variable) updated decoder state
        """
        x = torch.cat((self.embed(i), self.hx), dim=1)
        if s is not None and len(s[0]) == self.n_layers*2:
            s = list(s)
            for m in (0,1):
                ss = []
                for n in six.moves.range(0,len(s[m]),2):
                    ss.append(F.concat((s[m][n],s[m][n+1]), axis=1))
                s[m] = F.stack(ss, axis=0)

        if len(i) != 0:
            xs = torch.unsqueeze(x,0)
        else:
            xs = [x]

        if s is not None:
            dy, (hy, cy) = self.lstm(xs, (s[0], s[1]))
        else:
            dy, (hy, cy) = self.lstm( xs)

        return hy, cy, dy


    def predict(self, s):
        """Predict single-label log probabilities

        Args:
            s (any): Current (hidden, cell) states.
        Return:
            (~chainer.Variable) log softmax vector
        """
        y = self.out(self.proj(s[2][0]))
        return F.log_softmax(y, dim=1)

