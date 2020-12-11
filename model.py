import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

class CNNSingleLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, dilation_sizes, output_dim,
                 dropout, pad_idx):

        super().__init__()

        self.embedding_reduction = 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        #self.embedding.weight.requires_grad = False
        kernel_dimensions = zip(filter_sizes,dilation_sizes)


        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1,
                                              out_channels = n_filters,
                                              kernel_size = (fs, floor(embedding_dim/self.embedding_reduction)),
                                              dilation = (ds,1),
                                              padding = (int((fs*ds)/2),0)
                                             )

                                    for (fs,ds) in kernel_dimensions
                                    ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        #self.fc1 = nn.Linear(len(filter_sizes) * n_filters,len(filter_sizes)* n_filters)
        #self.fc2 = nn.Linear(len(filter_sizes) * n_filters,len(filter_sizes)* n_filters)

        #self.dropout = nn.Dropout(dropout)

        '''
        self.Conv1 = nn.Conv2d(in_channels=1,out_channels=n_filters,kernel_size=(3,1),padding=(1,0))
        self.Conv2 = nn.Conv2d(in_channels=n_filters,out_channels=n_filters,kernel_size=(3,1),padding=(1,0))
        self.Conv3 = nn.Conv2d(in_channels=n_filters,out_channels=n_filters,kernel_size=(3,1),padding=(1,0))
        #self.Conv4 = nn.Conv2d(in_channels=n_filters,out_channels=n_filters,kernel_size=(3,1),padding=(1,0))
        #self.Conv5 = nn.Conv2d(in_channels=n_filters,out_channels=n_filters,kernel_size=(3,1),padding=(1,0))
        self.Conv6 = nn.Conv2d(in_channels=n_filters,out_channels=1,kernel_size=(3,embedding_dim),padding=(1,0))
        '''


    def init_weights(self,val=1):
        for m in [self.convs,self.fc]:
            nn.init.orthogonal_(m.weight, val)
            torch.nn.utils.weight_norm(m)


    def forward(self, text):
        embedded = self.embedding(text)
        #embedded = self.dropout(embedded)
        #embedded = F.avg_pool1d(embedded,
         #   kernel_size = self.embedding_reduction,
         #   stride = self.embedding_reduction)
        embedded = embedded.unsqueeze(1)

        '''
        conved = self.Conv1(embedded)
        #conved = self.dropout(conved)
        conved = torch.relu(conved)
        #conved = F.avg_pool2d(conved,kernel_size=(3,1))
        conved = self.Conv2(conved)
        #conved = self.dropout(conved)
        conved = torch.relu(conved)
        #conved = F.avg_pool2d(conved,kernel_size=(3,1))
        conved = self.Conv3(conved)
        #conved = self.dropout(conved)
        conved = torch.relu(conved)
        #conved = F.avg_pool2d(conved,kernel_size=(3,1))
        #conved = self.Conv4(conved)
        #conved = torch.relu(conved)
        #conved = F.avg_pool2d(conved,kernel_size=(3,1))
        #conved = self.Conv5(conved)
        #conved = torch.relu(conved)
        #conved = F.avg_pool2d(conved,kernel_size=(3,1))
        conved = self.Conv6(conved)
        conved = torch.relu(conved)

        pooled = F.avg_pool1d(conved.squeeze(3), conved.shape[2]).squeeze(2)
        return pooled
        '''

        conved = [conv(embedded).squeeze(3) for conv in self.convs]
        pooled = [F.avg_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim = 1)
        linear = self.fc(cat)
        #linear = torch.mean(cat,dim=1,keepdim=True)
        return linear
