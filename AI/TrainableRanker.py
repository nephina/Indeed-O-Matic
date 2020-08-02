import pandas as pd
import numpy as np
import re
import random
import sys
import os
from PyQt5.QtWidgets import (QApplication, QDialog, QGridLayout, QPushButton, QTextEdit, QWidget, QHBoxLayout, QProgressBar, QLabel)
from trainer import Trainer


class Pairwise_Prompt(QDialog):
    def __init__(self, parent=None):
        super(Pairwise_Prompt, self).__init__(parent)


        self.LeftJobListing = QTextEdit()
        self.RightJobListing = QTextEdit()

        self.ListingPreference = ListingPreference
        self.LeftJobListing.setReadOnly(True)
        self.RightJobListing.setReadOnly(True)
        self.LeftSelectionButton = QPushButton('Select Job 1')
        self.RightSelectionButton = QPushButton('Select Job 2')
        self.LeftSelectionButton.setDefault(True)
        self.RightSelectionButton.setDefault(True)
        self.ProgressBar = QProgressBar()
        self.ProgressBar.setRange(0, 50)
        self.ProgressBar.setValue(0)
        self.StatusText = QLabel('Currently in: user entry mode')

        topLayout = QHBoxLayout()
        mainLayout = QGridLayout()
        mainLayout.addLayout(topLayout, 4, 2, 1, 1)
        mainLayout.addWidget(self.LeftJobListing, 0, 0)
        mainLayout.addWidget(self.RightJobListing, 0, 1)
        mainLayout.addWidget(self.LeftSelectionButton,1,0)
        mainLayout.addWidget(self.RightSelectionButton,1,1)
        mainLayout.addWidget(self.ProgressBar,3,0,1,2)
        mainLayout.addWidget(self.StatusText,2,0,1,2)
        self.setLayout(mainLayout)

        self.setWindowTitle("AI training example generation")


        self.LeftSelectionButton.clicked.connect(Left_Pref)
        self.RightSelectionButton.clicked.connect(Right_Pref)


def Read_Listings():
    Listings = pd.read_csv('Data/Listings.csv') #Read all listings
    Listings_len = len(Listings)
    ListingsCleaned = Listings.copy()

    for column_to_clean in ['Position','Company','Location','Salary','Summary']:
        ListingsCleaned[column_to_clean].replace('\n',' ', regex=True,inplace=True) #Replace all newline char's with spaces
        ListingsCleaned[column_to_clean].fillna('', inplace=True) #Replaces all NaN's with blank strings

        # remove all unusual text characters in the text and turn everything lowercase to reduce dictionary size
        cleaningfunction = lambda piece_of_text: re.sub(r'\.(?=[^ \W\d])', '. ',piece_of_text[column_to_clean].lower())
        ListingsCleaned[column_to_clean] = ListingsCleaned.apply(cleaningfunction,'columns')

        DescriptionAndRank = pd.DataFrame({'Description': ListingsCleaned['Position'] + ' ' + ListingsCleaned['Company'] + ' ' + ListingsCleaned['Location'] + ' ' + ListingsCleaned['Salary'] + ' ' + ListingsCleaned['Summary'],'Rating': ListingsCleaned['Rating']})
        DescriptionAndRank['SortKey'] = range(len(DescriptionAndRank))
        DescriptionAndRank.to_csv('Data/DescriptionAndRank.csv',index=False)

    return Listings, Listings_len, ListingsCleaned, DescriptionAndRank

def Update_Single_Listing(preference):
    New_listing,listing_index = Get_Random_Job()
    if preference == 1: #if the right job was preferred
        window.LeftJobListing.setPlainText(
            'Position: '+New_listing['Position']+'\n'+
            'Company: '+New_listing['Company']+'\n'+
            'Location: '+New_listing['Location']+'\n'+
            'Salary: '+str(New_listing['Salary'])+'\n'+
            'Company rating: '+str(New_listing['Company rating'])+'\n'+
            'Summary: '+New_listing['Summary']+'\n'
            )
    if preference == 0: #if the left job was preferred
        window.RightJobListing.setPlainText(
            'Position: '+New_listing['Position']+'\n'+
            'Company: '+New_listing['Company']+'\n'+
            'Location: '+New_listing['Location']+'\n'+
            'Salary: '+str(New_listing['Salary'])+'\n'+
            'Company rating: '+str(New_listing['Company rating'])+'\n'+
            'Summary: '+New_listing['Summary']+'\n'
            )
    return (listing_index)

def Update_Both_Listings(close_pair=False):
    def Write_Both_to_App(Left_listing,Right_listing):
        window.LeftJobListing.setPlainText(
            'Position: '+str(Left_listing['Position'])+'\n'
           +'Company: '+str(Left_listing['Company'])+'\n'
           +'Location: '+str(Left_listing['Location'])+'\n'
           +'Salary: '+str(Left_listing['Salary'])+'\n'
           +'Company rating: '+str(Left_listing['Company rating'])+'\n'
           +'Summary: '+str(Left_listing['Summary'])+'\n'
            )
        window.RightJobListing.setPlainText(
            'Position: '+str(Right_listing['Position'])+'\n'
           +'Company: '+str(Right_listing['Company'])+'\n'
           +'Location: '+str(Right_listing['Location'])+'\n'
           +'Salary: '+str(Right_listing['Salary'])+'\n'
           +'Company rating: '+str(Right_listing['Company rating'])+'\n'
           +'Summary: '+str(Right_listing['Summary'])+'\n'
            )
    if close_pair:
        Left_listing,Right_listing,listing_indices = Get_Close_Pair()
        Write_Both_to_App(Left_listing,Right_listing)
    else:
        Left_listing,Right_listing,listing_indices = Get_Random_Job_Pair()
        Write_Both_to_App(Left_listing,Right_listing)

    return listing_indices

def Run_AI_Training():
    global Listings, Listings_len, ListingsCleaned, DescriptionAndRank
    FinalRankedPairs = pd.DataFrame(Pairwise_Ranked_Listings)
    FinalRankedPairs.columns = ['Description','Rating']
    FinalRankedPairs['SortKey'] = range(len(FinalRankedPairs))

    FinalRankedPairs.to_csv('Data/RankedPairs.csv',index=False)
    Trainer(window, Listings, epochs = 20)
    window.ProgressBar.setRange(0, 50)
    window.ProgressBar.setValue(0)
    window.StatusText.setText('Currently in: user entry mode')
    Listings, Listings_len, ListingsCleaned, DescriptionAndRank = Read_Listings()
    InitialDataCollectStep[0] = 0


def Left_Pref():
    global listing_indices, Pairwise_Ranked_Listings
    if InitialDataCollectStep[0] == 1:
        TotalSelectionsCount[0] += 1
        window.ProgressBar.setValue(TotalSelectionsCount[0])
        RightSelectCount[0] = 0
        LeftSelectCount[0] += 1
        if LeftSelectCount[0] < 10:
            right_listing_index = Update_Single_Listing(0)
            listing_indices[1] = right_listing_index
        else:
            left_listing_index = Update_Single_Listing(1)
            listing_indices[0] = left_listing_index
            LeftSelectCount[0] = 0

        Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[0]],1])
        Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[1]],-1])
    else:
        TotalSelectionsCount[0] += 1
        window.ProgressBar.setValue(TotalSelectionsCount[0])
        listing_indices = Update_Both_Listings(close_pair=True)

        Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[0]],1])
        Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[1]],-1])


    if TotalSelectionsCount[0] >= 50:
        Run_AI_Training()
        TotalSelectionsCount[0] = 0 #reset the counter

def Right_Pref():
    global listing_indices, Pairwise_Ranked_Listings
    if InitialDataCollectStep[0] == 1:
        TotalSelectionsCount[0] += 1
        window.ProgressBar.setValue(TotalSelectionsCount[0])
        LeftSelectCount[0] = 0
        RightSelectCount[0] += 1
        if RightSelectCount[0] < 10:
            left_listing_index = Update_Single_Listing(1)
            listing_indices[0] = left_listing_index

        else:
            right_listing_index = Update_Single_Listing(0)
            listing_indices[1] = right_listing_index
            RightSelectCount[0] = 0

        Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[0]],-1])
        Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[1]],1])

    else:
        TotalSelectionsCount[0] += 1
        window.ProgressBar.setValue(TotalSelectionsCount[0])
        listing_indices = Update_Both_Listings(close_pair=True)

        Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[0]],1])
        Pairwise_Ranked_Listings.append([DescriptionAndRank['Description'][listing_indices[1]],-1])

    if TotalSelectionsCount[0] >= 50:
        print('AI time')
        Run_AI_Training()
        TotalSelectionsCount[0] = 0 #reset the counter




def Get_Random_Job():
    listing_index = random.sample(range(0,Listings_len-1),2)
    Listing = Listings.iloc[listing_index[0]]
    return Listing,listing_index[0]

def Get_Random_Job_Pair():
    listing_pair_indices = random.sample(range(0,Listings_len-1),2)
    Listing1 = Listings.iloc[listing_pair_indices[0]]
    Listing2 = Listings.iloc[listing_pair_indices[1]]
    return Listing1,Listing2,listing_pair_indices

def Get_Close_Pair():
    Listings.sort_values(by=['Rating'],inplace=True,ascending=False)
    deltas = abs(np.diff(Listings['Rating']))
    deltas = [deltas.tolist()]
    deltas.append([index for index in range(len(deltas[0]))])
    deltas = pd.DataFrame(deltas)
    deltas.sort_values(by=[0],inplace=True,axis=1)
    deltas = deltas.transpose()
    deltas = deltas.reset_index(drop=True)
    small_delta_random_sample_index = random.sample(range(0,int(Listings_len*0.1)),1)
    listing_indices = [int(deltas[1][small_delta_random_sample_index]), int(deltas[1][small_delta_random_sample_index])+1]
    return Listings.iloc[listing_indices[0]],Listings.iloc[listing_indices[1]],listing_indices




ListingPreference=[]
Pairwise_Ranked_Listings = []
LeftSelectCount = [0]
RightSelectCount = [0]
TotalSelectionsCount = [0]
InitialDataCollectStep = [1]

app = QApplication(sys.argv)
window = Pairwise_Prompt()
window.show()

Listings, Listings_len, ListingsCleaned, DescriptionAndRank = Read_Listings()
listing_indices = Update_Both_Listings(close_pair=False)

app.exec_()
