import torch, torchtext, nltk
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from math import ceil
import pandas as pd
import re
from model import CNNSingleLayer

def train(model, iterator, optimizer, criterion, loss_type = 'order'):

    epoch_loss = 0
    epoch_order_loss = 0
    epoch_centered_normal_loss = 0
    epoch_beta_dist_loss = 0
    epoch_stdev_loss = 0
    epoch_acc = 0
    epoch_mean = 0
    epoch_std = 0

    model.train()
    for i,batch in enumerate(iterator):
        predictions = model(batch.Description[0]).squeeze(1)
        order_loss = same_order_loss(predictions,batch.Rank)

        if loss_type == 'order':
            loss = order_loss

        elif loss_type == 'orderandstdev':
            stdev_loss = (1-torch.std(predictions)).pow(2)
            if torch.isnan(stdev_loss):
                loss = order_loss
            else:
                loss = order_loss+stdev_loss

        elif loss_type == 'orderandcenterednormal':
            centered_normal_loss = criterion(torch.sort(predictions).values,torch.sort(torch.randn_like(predictions)).values)
            if torch.isnan(centered_normal_loss).any():
                loss = order_loss
            else:
                loss = order_loss+centered_normal_loss

        elif loss_type == 'orderandbeta':
            distribution = torch.distributions.beta.Beta(torch.tensor([1.2],dtype=torch.float32),torch.tensor([4],dtype=torch.float32)).sample(predictions.size()).squeeze(1)
            distribution = 1-distribution
            beta_dist_loss = criterion(torch.sort(predictions).values,torch.sort(distribution).values)
            if torch.isnan(beta_dist_loss).any():
                loss = order_loss
            else:
                loss = order_loss+beta_dist_loss


        loss.backward()
        optimizer.step()
        model.zero_grad()
        '''
        if (i+1) % 10 == 0:      # Wait for several backward steps
            for gp in model.parameters():
                if gp.grad is not None:
                    gp.grad = gp.grad/10
            optimizer.step()               # Now we can do an optimizer step
            model.zero_grad()
        '''
        epoch_mean += torch.mean(predictions).item()
        epoch_std += torch.std(predictions).item()
        epoch_order_loss += order_loss.item()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator), epoch_order_loss / len(iterator), epoch_mean / len(iterator), epoch_std / len(iterator)

def test(window,model, iterator):

    epoch_loss = 0
    epoch_acc = 0
    test_result = []
    model.eval()
    with torch.no_grad():
        for i,batch in enumerate(iterator):
            predictions = model(batch.Description[0]).squeeze(1)
            for descript,loc,predict in zip(batch.Description[0],batch.SortKey,predictions):
                test_result.append([descript.numpy(),predict.item(),int(loc.item())])
            window.ProgressBar.setValue(i+1)

    return test_result


def same_order_loss(output,target):
        reshaped_prediction = output.view(torch.div(len(output),2),2)
        reshaped_target = target.view(torch.div(len(target),2),2)
        #This reshaping takes the pairwise batch data in format: [[example1left][example1right][example2left][example2right]] and reshapes it to [[example1left,example1right],[example2left,example2right]]

        x = -((reshaped_prediction[:,0]-reshaped_prediction[:,1])*(reshaped_target[:,0]-reshaped_target[:,1])) #This function takes all of the pairwise comparisons and outputs a positive num if wrong order, negative num if correct order

        loss = torch.mean(torch.relu(x))#This function punishes any incorrectly ranked items, while allowing correctly-ranked items to move around at will in the ranking system.
        return(loss)

def trainer(window, Listings):
    window.StatusText.setText('Building Training Dataset')
    window.ProgressBar.setRange(0, 100)
    window.ProgressBar.setValue(0)
    DescriptionField = torchtext.data.Field(
    sequential=True, #words have order, sequence matters
    include_lengths=True, #batching function tries to batch similar-length lines together

    # NLTK recognizes hyphenated bigrams and understands mis-spelled words,
    # it takes our long string of text and breaks it into individual words or "tokens",
    # fixing it along the way. The default would have been a split() function.
    tokenize=nltk.tokenize.word_tokenize,
    use_vocab=True, # we're going to use the GloVe vocabulary vectorizer to turn tokens into integers
    batch_first = True #batch dimension comes first in the tensor
    )

    RankField = torchtext.data.Field(
    sequential=False,
    tokenize=None,
    include_lengths=None,
    use_vocab=None,
    batch_first = True,

    #default is torch.long, fine for integer representations of words but not 0-1 ranking
    dtype=torch.float
    )


    # This needs to be kept with the data. The first reason is that the pairs of positive and negative pairwise ranked examples need to be kept in exactly the same order and the iterator doesn't care what order it loads them in unless you tell it. The second is that we can't just write the big chunk of words that have been cleaned, lowercased, and tokenized to a new file, we want to apply a rank to the original text data in all its unformatted, messy glory. So we need to know what rank in the cleaned and tokenized data went to which listing in the original job listing plaintext. This also means we're speeding up the process since we don't need to convert the vectorized token descriptions back into plaintext.

    SortField = torchtext.data.Field(
    sequential=False,
    tokenize=None,
    include_lengths=None,
    use_vocab=None,
    batch_first = True,
    dtype=torch.float)

    Fields = [('Description', DescriptionField),('Rank', RankField),('SortKey', SortField)]


    dataset = torchtext.data.TabularDataset('Data/DescriptionAndRank.csv','CSV',skip_header=True,fields = Fields)
    trainset = torchtext.data.TabularDataset('Data/RankedPairs.csv','CSV',skip_header=True,fields = Fields)
    window.ProgressBar.setValue(80)

    DescriptionField.build_vocab(dataset, max_size = 20000, min_freq = 5) #max 30,000 words/tokens
    print('Unique tokens in Description vocabulary: {}'.format(len(DescriptionField.vocab)))
    print(DescriptionField.vocab.itos[-100:]) #print the least popular 100 tokens (the first two are the "unknown" and the "padding" tokens)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    window.ProgressBar.setValue(90)

    train_iterator = torchtext.data.Iterator(
        trainset,
        batch_size = len(trainset), #MUST BE A MULTIPLE OF 2
        device = device,
        sort_key = lambda x: int(x.SortKey),
        sort=True
        ) #sort it by the sortkey; in other words, tell it to not shuffle them at all

    test_iterator = torchtext.data.Iterator(
        dataset,
        batch_size = 100,
        device = device,
        sort_key=lambda x: len(x.Description)
        )#sort by the length of the job description, that way we group
        # similar-length job descriptions together, avoiding having too much padding on the ends


    #Define the net characteristics

    INPUT_DIM = len(DescriptionField.vocab)
    EMBEDDING_DIM = 1 # when we turned our tokens into vectors, this was the length of the vector
    N_FILTERS =  1#ceil(len(trainset)/500) # how many features the convolutions learn
    FILTER_SIZES = [1]
    DILATION_SIZES = [1]
    OUTPUT_DIM = 1
    DROPOUT = 0.5

    UNK_IDX = DescriptionField.vocab.stoi[DescriptionField.unk_token]
    PAD_IDX = DescriptionField.vocab.stoi[DescriptionField.pad_token]

    model = CNNSingleLayer(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, DILATION_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

    pretrained_embeddings = DescriptionField.vocab.vectors

    #model.embedding.weight.data.copy_(pretrained_embeddings)

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    try:
        model.load_state_dict(torch.load('RankPrediction-model.pkl'))
    except:
        print('no previously existing trained model')
    model = model.to(device)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad == True],lr=0.1)

    criterion = nn.MSELoss()#KLDivLoss(reduction='batchmean')
    criterion.to(device)


    def rerank_jobs(window):
        window.StatusText.setText('Ranking Jobs')
        window.ProgressBar.setRange(0, len(test_iterator))
        window.ProgressBar.setValue(0)

        test_result = test(window,model, test_iterator)

        PredictedPlaintextListingsandRanks = []


        window.StatusText.setText('Writing Ranked Jobs')
        window.ProgressBar.setRange(0, len(test_result))
        window.ProgressBar.setValue(0)

        # Rank all the jobs in the original listing text

        #Re-order the rankings from the prediction step
        test_result_df = pd.DataFrame(test_result)
        test_result_df.sort_values(by=[2],inplace=True)
        test_result_df.reset_index(drop=True)
        Listings['Rating']= test_result_df[1].tolist()

        PredictedPlaintextListingsandRanks = []
        #for i,listing in enumerate(test_result):
        #    window.ProgressBar.setValue(i+1)
        #    location = int(listing[2].item())
        #    Listings['Rating'][location] = listing[1]

        #Re-order the AI-ranked jobs
        tempListings = pd.DataFrame(Listings)
        tempListings.sort_values(by=['Rating'],inplace=True, ascending=False)
        tempListings.reset_index(drop=True)
        tempListings.to_csv('Listings.csv',index=False)
        return window


    window.StatusText.setText('Training the Neural Net')
    window.ProgressBar.setRange(0, 10*N_FILTERS)
    window.ProgressBar.setValue(0)
    epoch = 0
    train_loss = 10e10
    train_order_loss = 10e10
    train_std = 0

    while (train_order_loss != 0) or (1-train_std > 0.01):

        # resort the paired examples to mix up the data
        sort_key = [x for x in range(len(trainset))]
        sort_key = np.reshape(sort_key,(-1,2))
        np.random.shuffle(sort_key)
        sort_key = np.reshape(sort_key,(-1))
        for i in range(int(len(trainset)/2)):
            trainset.examples[(2*i)].SortKey = sort_key[2*i]
            trainset.examples[(2*i)+1].SortKey = sort_key[(2*i)+1]
        train_iterator = torchtext.data.Iterator(
        trainset,
        batch_size = len(trainset), #MUST BE A MULTIPLE OF 2
        device = device,
        sort_key = lambda x: int(x.SortKey),
        sort=True)

        train_loss, train_order_loss, train_mean, train_std = train(model, train_iterator, optimizer, criterion,loss_type='orderandstdev')
        print(train_loss/(train_std+1.0e-25),train_order_loss,train_mean,train_std)

        torch.save(model.state_dict(), 'RankPrediction-model.pkl')
        epoch += 1
        if epoch % 50 == 0:
            print('Reranking jobs')
            window  = rerank_jobs(window)
            window.StatusText.setText('Training the Neural Net')
            window.ProgressBar.setRange(0, 10*N_FILTERS)
            window.ProgressBar.setValue(0)
        #window.ProgressBar.setValue(epoch)\

    rerank_jobs(window)
    '''
    train_iterator = torchtext.data.Iterator(
        trainset,
        batch_size = len(trainset), #MUST BE A MULTIPLE OF 2
        device = device,
        sort_key = lambda x: int(x.SortKey),
        sort=True
        ) #sort it by the sortkey; in other words, tell it to not shuffle them at all
    train_order_loss = 10e10

    while train_order_loss != 0:# and epoch < 10*N_FILTERS:
        train_loss, train_order_loss, train_mean, train_std = train(model, train_iterator, optimizer, criterion,loss_type='orderandstdev')
        print(train_loss,train_order_loss,train_mean,train_std)

        torch.save(model.state_dict(), 'RankPrediction-model.pkl')
        epoch += 1
        if epoch % 50 == 0:
            print('Reranking jobs')
            window  = rerank_jobs(window)
            window.StatusText.setText('Training the Neural Net')
            window.ProgressBar.setRange(0, 10*N_FILTERS)
            window.ProgressBar.setValue(0)
        #window.ProgressBar.setValue(epoch)

    rerank_jobs(window)
    '''
