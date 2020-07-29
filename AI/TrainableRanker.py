import torch, torchtext, nltk
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import re, string
from tqdm import tqdm
from tqdm import notebook
import random
import sys
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QDialog, QGridLayout, QPushButton, QTextEdit, QWidget, QHBoxLayout)

Listings = pd.read_csv('Data/Listings.csv')
Listings_len = len(Listings)


for column_to_clean in ['Position','Company','Location','Salary','Summary']:
    Listings[column_to_clean].replace('\n',' ', regex=True,inplace=True) #Replace all newline char's with spaces
    Listings[column_to_clean].fillna('', inplace=True) #Replaces all NaN's with blank strings

    # remove all unusual text characters in the text and turn everything lowercase to reduce dictionary size
    cleaningfunction = lambda piece_of_text: re.sub(r'\.(?=[^ \W\d])', '. ',piece_of_text[column_to_clean].lower())
    Listings[column_to_clean] = Listings.apply(cleaningfunction,'columns')

DescriptionAndRank = pd.DataFrame({'Description': Listings['Position'] + ' ' + Listings['Company'] + ' ' + Listings['Location'] + ' ' + Listings['Salary'] + ' ' + Listings['Summary'],'Rating': Listings['Rating']})


class AI_Training_Example_Prompt(QDialog):
    def __init__(self, parent=None):
        super(AI_Training_Example_Prompt, self).__init__(parent)


        self.LeftJobListing = QTextEdit()
        self.RightJobListing = QTextEdit()

        self.ListingPreference = ListingPreference
        self.LeftJobListing.setReadOnly(True)
        self.RightJobListing.setReadOnly(True)
        self.LeftSelectionButton = QPushButton('Select Job 1')
        self.RightSelectionButton = QPushButton('Select Job 2')
        self.LeftSelectionButton.setDefault(True)
        self.RightSelectionButton.setDefault(True)

        topLayout = QHBoxLayout()
        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 2, 2, 1, 1)
        mainLayout.addWidget(self.LeftJobListing, 0, 0)
        mainLayout.addWidget(self.RightJobListing, 0, 1)
        mainLayout.addWidget(self.LeftSelectionButton,1,0)
        mainLayout.addWidget(self.RightSelectionButton,1,1)
        self.setLayout(mainLayout)

        self.setWindowTitle("AI training example generation")


        self.LeftSelectionButton.clicked.connect(LeftPref)
        self.RightSelectionButton.clicked.connect(RightPref)

def UpdateListings():
    Left_listing,Right_listing,listing_pair_indices = GetRandomJobPair()
    window.LeftJobListing.setPlainText(
        'Position: '+Left_listing['Position']+'\n'
       +'Company: '+Left_listing['Company']+'\n'
       +'Location: '+Left_listing['Location']+'\n'
       +'Salary: '+str(Left_listing['Salary'])+'\n'
       +'Company rating: '+str(Left_listing['Company rating'])+'\n'
       +'Summary: '+Left_listing['Summary']+'\n'
        )
    window.RightJobListing.setPlainText(
        'Position: '+Right_listing['Position']+'\n'
       +'Company: '+Right_listing['Company']+'\n'
       +'Location: '+Right_listing['Location']+'\n'
       +'Salary: '+str(Right_listing['Salary'])+'\n'
       +'Company rating: '+str(Right_listing['Company rating'])+'\n'
       +'Summary: '+Right_listing['Summary']+'\n'
        )
    return (listing_pair_indices)


ListingPreference=[]
Pairwise_Ranked_Listings = []

def LeftPref():
    listing_indices = UpdateListings()
    Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[0]],1])
    Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[1]],-1])
    #print(pd.DataFrame(Pairwise_Ranked_Listings))
def RightPref():
    listing_indices = UpdateListings()
    Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[0]],-1])
    Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[1]],1])
    #print(pd.DataFrame(Pairwise_Ranked_Listings))


def GetRandomJobPair():
    listing_pair_indices = random.sample(range(0,Listings_len-1),2)
    Listing1 = Listings.iloc[listing_pair_indices[0]]
    Listing2 = Listings.iloc[listing_pair_indices[1]]
    return[Listing1,Listing2,listing_pair_indices]


app = QApplication(sys.argv)
window = AI_Training_Example_Prompt()
window.show()
UpdateListings()
app.exec_()
FinalRankedPairs = pd.DataFrame(Pairwise_Ranked_Listings)
FinalRankedPairs.columns = ['Description','Rating']
print(FinalRankedPairs)
FinalRankedPairs.to_csv('Data/FinalRankedPairs.csv',index=False)
