#!/usr/bin/env python
"""Multimodal sequence encoder 
   Copyright 2016 Mitsubishi Electric Research Labs
"""

import math
import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F


class MMEncoder(nn.Module):

    def __init__(self, in_size, out_size, enc_psize=[], enc_hsize=[], att_size=100,
                 state_size=100):
        if len(enc_psize)==0:
            enc_psize = in_size
        if len(enc_hsize)==0:
            enc_hsize = [0] * len(in_size)

        # make links
        super(MMEncoder, self).__init__()
        # memorize sizes
        self.n_inputs = len(in_size)
        self.in_size = in_size
        self.out_size = out_size
        self.enc_psize = enc_psize
        self.enc_hsize = enc_hsize
        self.att_size = att_size
        self.state_size = state_size
        # encoder
        self.l1f_x = nn.ModuleList()
        self.l1f_h = nn.ModuleList()
        self.l1b_x = nn.ModuleList()
        self.l1b_h = nn.ModuleList()
        self.emb_x = nn.ModuleList()
        for m in six.moves.range(len(in_size)):
            self.emb_x.append(nn.Linear(self.in_size[m], self.enc_psize[m]))
            if enc_hsize[m] > 0:
                self.l1f_x.append(nn.Linear(enc_psize[m], 4 * enc_hsize[m]))
                self.l1f_h.append(nn.Linear(enc_hsize[m], 4 * enc_hsize[m], bias=False))
                self.l1b_x.append(nn.Linear(enc_psize[m], 4 * enc_hsize[m]))
                self.l1b_h.append(nn.Linear(enc_hsize[m], 4 * enc_hsize[m], bias=False))
        # temporal attention
        self.atV = nn.ModuleList()
        self.atW = nn.ModuleList()
        self.atw = nn.ModuleList()
        self.lgd = nn.ModuleList()
        for m in six.moves.range(len(in_size)):
            enc_hsize_ = 2 * enc_hsize[m] if enc_hsize[m] > 0 else enc_psize[m]
            self.atV.append(nn.Linear(enc_hsize_, att_size))
            self.atW.append(nn.Linear(state_size, att_size))
            self.atw.append(nn.Linear(att_size, 1))
            self.lgd.append(nn.Linear(enc_hsize_, out_size))

    # Make an initial state
    def make_initial_state(self, hiddensize):
        return {name: torch.zeros(self.bsize, hiddensize, dtype=torch.float)
                for name in ('c1', 'h1')}

    # Encoder functions
    def embed_x(self, x_data, m):
        x0 = [x_data[i]
              for i in six.moves.range(len(x_data))]
        return self.emb_x[m](torch.cat(x0, 0).cuda().float())

    def forward_one_step(self, x, s, m):
        x_new = x + self.l1f_h[m](s['h1'].cuda())
        x_list = torch.split(x_new, self.enc_hsize[m], dim=1)
        x_list = list(x_list)
        c1 = torch.tanh(x_list[0]) * F.sigmoid(x_list[1]) + s['c1'].cuda() * F.sigmoid(x_list[2])
        h1 = torch.tanh(c1) * F.sigmoid(x_list[3])
        return {'c1': c1, 'h1': h1}

    def backward_one_step(self, x, s, m):
        x_new = x + self.l1b_h[m](s['h1'].cuda())
        x_list = torch.split(x_new, self.enc_hsize[m], dim=1)
        x_list = list(x_list)
        c1 = torch.tanh(x_list[0]) * F.sigmoid(x_list[1]) + s['c1'].cuda() * F.sigmoid(x_list[2])
        h1 = torch.tanh(c1) * F.sigmoid(x_list[3])
        return {'c1': c1, 'h1': h1}

    # Encoder main
    def encode(self, x):
        h1 = [None] * self.n_inputs
        for m in six.moves.range(self.n_inputs):
            # self.emb_x=self.__dict__['emb_x%d' % m]
            if self.enc_hsize[m] > 0:
                # embedding
                seqlen = len(x[m])
                h0 = self.embed_x(x[m], m)
                # forward path
                aa = self.l1f_x[m](F.dropout(h0, training=self.train))
                fh1 = torch.split(self.l1f_x[m](F.dropout(h0, training=self.train)), self.bsize, dim=0)
                fstate = self.make_initial_state(self.enc_hsize[m])
                h1f = []
                for h in fh1:
                    fstate = self.forward_one_step(h, fstate, m)
                    h1f.append(fstate['h1'])
                # backward path
                bh1 = torch.split(self.l1b_x[m](F.dropout(h0, training=self.train)), self.bsize, dim=0)
                bstate = self.make_initial_state(self.enc_hsize[m])
                h1b = []
                for h in reversed(bh1):
                    bstate = self.backward_one_step(h, bstate, m)
                    h1b.insert(0, bstate['h1'])
                # concatenation
                h1[m] = torch.cat([torch.cat((f, b), 1)
                                   for f, b in six.moves.zip(h1f, h1b)], 0)
            else:
                # embedding only
                h1[m] = torch.tanh(self.embed_x(x[m], m))
        return h1

    # Attention
    def attention(self, h, vh, s):
        c = [None] * self.n_inputs

        for m in six.moves.range(self.n_inputs):
            bsize = self.bsize
            seqlen = h[m].data.shape[0] / bsize
            csize = h[m].data.shape[1]
            asize = self.att_size

            ws = self.atW[m](s)
            vh_m = vh[m].view(seqlen, bsize, asize)
            e1 = vh_m + ws.expand_as(vh_m)
            e1 = e1.view(seqlen * bsize, asize)
            e = torch.exp(self.atw[m](torch.tanh(e1)))
            e = e.view(seqlen, bsize)
            esum = e.sum(0)
            e = e / esum.expand_as(e)
            h_m = h[m].view(seqlen, bsize, csize)
            h_m = h_m.permute(2,0,1)
            c_m = h_m * e.expand_as(h_m)
            c_m = c_m.permute(1,2,0)
            c[m] = c_m.mean(0)
        return c

    # Simple modality fusion
    def simple_modality_fusion(self, c, s):
        g = 0.
        for m in six.moves.range(self.n_inputs):
            g += self.lgd[m](F.dropout(c[m]))
        return g

    # forward propagation routine
    def __call__(self, s, x, train=True):
        self.bsize = x[0][0].shape[0]
        h1 = self.encode(x)
        vh1 = [self.atV[m](h1[m]) for m in six.moves.range(self.n_inputs)]
        # attention
        c = self.attention(h1, vh1, s)
        g = self.simple_modality_fusion(c, s)
        return torch.tanh(g)

