import torch, torchtext, nltk
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from tqdm import notebook
from model import CNN

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_order_loss = 0
    epoch_centered_normal_loss = 0
    epoch_acc = 0

    model.train()
    for i,batch in enumerate(iterator):
        optimizer.zero_grad()
        predictions = model(batch.Description[0]).squeeze(1)

        order_loss = same_order_loss(predictions,batch.Rank)
        centered_normal_loss = criterion(predictions,(torch.randn_like(predictions))).pow(2)
        loss = order_loss+centered_normal_loss
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
        epoch_order_loss += order_loss.item()
        epoch_centered_normal_loss += centered_normal_loss.item()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator), epoch_order_loss / len(iterator), epoch_centered_normal_loss / len(iterator)

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
        x = -((output[0]-output[1])*(target[0]-target[1])) #positive num if wrong order, negative num if correct order
        loss = (x*torch.sigmoid(x))*torch.exp(x) #SiLU times e^x provides high motivation to make sure all rankings are in the correct order or at least close to it, but also pushes the ones in the correct order a little bit apart as well. This outward pressure on the ranking distribution is countered by the KL Divergence loss on a normal distribution, keeping the distribution centered and pressuring it to come down to a variance of 1
        return(loss)

def Trainer(window, Listings, epochs=10,features=10):
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


    # This needs to be kept with the data. The first reason is that the pairs of positive and negative pairwise ranked examples need to be kept in exactly the same order and the iterator doesn't care what order it loads them in unless you tell it. The second is that we can't just write the big chunk of words that have been cleaned, lowercased, and tokenized to a new file, we want to apply a rank to the original text data. So we need to know what rank in the cleaned and tokenized data went to which listing in the original job listing plaintext. This also helps to speed up the process since we don't need to convert the vectorized words back into text.

    SortField = torchtext.data.Field(
    sequential=False,
    tokenize=None,
    include_lengths=None,
    use_vocab=None,
    batch_first = True,

    #default is torch.long, fine for integer representations of words but not 0-1 ranking
    dtype=torch.float)

    Fields = [('Description', DescriptionField),('Rank', RankField),('SortKey', SortField)]


    dataset = torchtext.data.TabularDataset('Data/DescriptionAndRank.csv','CSV',skip_header=True,fields = Fields)
    trainset = torchtext.data.TabularDataset('Data/RankedPairs.csv','CSV',skip_header=True,fields = Fields)
    window.ProgressBar.setValue(80)

    '''
    The beauty of the GloVe vocabulary model is that it has been trained to "group" certain words with other words
    so that meaning is approximated in numerical format. We are using the 50-dimensional version, which means there
    are 50 dimensions in which one word can be "similar" to any other word. So for instance, if the word 'dog' were
    represented in its 34th dimension with a floating point of 0.893833, you would expect to find that the
    34th dimension of the word 'puppy' was close to that numerical value. In this way, the neural network can learn to
    approximate meaning, without having to define absurdly complex almost step-wise functions for randomly-assigned
    word vectors.
    '''
    DescriptionField.build_vocab(dataset, max_size = 40000, vectors='glove.6B.50d') #max 30,000 words/tokens
    #print('Unique tokens in Description vocabulary: {}'.format(len(DescriptionField.vocab)))
    #print(DescriptionField.vocab.itos[2:102]) #print the most popular 100 tokens (the first two are the "unknown" and the "padding" tokens)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    window.ProgressBar.setValue(90)

    train_iterator = torchtext.data.Iterator(
        trainset,
        batch_size = 2,
        device = device,
        sort_key = lambda x: int(x.SortKey),
        sort=True
        #sort_key=lambda x: len(x.Description)
        )#sort by the length of the job description, that way we group
                            # similar-length job descriptions together, avoiding having too much padding on the ends

    test_iterator = torchtext.data.Iterator(
        dataset,
        batch_size = 20,
        device = device,
        sort_key=lambda x: len(x.Description)
        )#sort by the length of the job description, that way we group
        # similar-length job descriptions together, avoiding having too much padding on the ends


    #Define the net characteristics

    INPUT_DIM = len(DescriptionField.vocab)
    EMBEDDING_DIM = 50 # when we turned our tokens into vectors, this was the length of the vector
    N_FILTERS = features # how many features the convolutions learn
    FILTER_SIZES = [2,3,4,5]#,5,5,5,5,5,5]
    DILATION_SIZES = [1,1,1,1]#,2,4,8,16,32,64]
    OUTPUT_DIM = 1
    DROPOUT = 0.0
    PAD_IDX = DescriptionField.vocab.stoi[DescriptionField.pad_token]

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, DILATION_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

    pretrained_embeddings = DescriptionField.vocab.vectors

    model.embedding.weight.data.copy_(pretrained_embeddings)

    UNK_IDX = DescriptionField.vocab.stoi[DescriptionField.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    import torch.optim as optim

    optimizer = optim.Adam(model.parameters())
    try:
        model.load_state_dict(torch.load('RankPrediction-model.pkl'))
    except:
        print('no previously existing trained model')
    model = model.to(device)
    criterion = nn.KLDivLoss(reduction='batchmean')
    criterion.to(device)

    window.StatusText.setText('Training the Neural Net')
    window.ProgressBar.setRange(0, epochs)
    window.ProgressBar.setValue(0)

    for epoch in range(epochs):

        train_loss, train_order_loss, train_centered_loss = train(model, train_iterator, optimizer, criterion)
        print(train_loss,train_order_loss,train_centered_loss)
        torch.save(model.state_dict(), 'RankPrediction-model.pkl')
        window.ProgressBar.setValue(epoch+1)

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
    tempListings.to_csv('Data/Listings.csv',index=False)
