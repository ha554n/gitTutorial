import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from math import pi
from itertools import combinations_with_replacement
import pickle
from IPython.display import display
# pd.set_option('display.max_rows', 500)
# pd.options.display.max_colwidth=100
from natsort import natsorted
import seaborn as sns
import scipy.stats as stats 
from collections import Counter
from itertools import groupby
from operator import itemgetter
from itertools import combinations
import time
import math as math
import inspect





#%% helper functions
def atoi(text):
  '''
  returns int parts of strings
  '''
  return int(text) if text.isdigit() else text

def natural_keys(text):
  '''
  alist.sort(key=natural_keys) sorts in human order
  http://nedbatchelder.com/blog/200712/human_sorting.html
  (See Toothy's implementation in the comments)
  '''
  return [ atoi(c) for c in re.split(r'(\d+)', text) ]
#%% subject class no simulation

class Subject():
    """
    this class is used to load subjets as 'objects', this allows
    various aspects to be set as object attributes
    """
    def __init__(self,df,subname):
        
        self.subname=subname #subnum
        self.df=pd.read_csv(df)
        self.prolificID=self.df['prolificID'][0]# this is the unique prolific ID identifier
        self.Date=self.df.date[0]#date/time of access
        #read in the conditions because pavlovia is not giving the conditions anymore, these contain the actual order of comparisons
        self.ShirtsIdx=pd.read_csv(r"/content/drive/My Drive/GU Neuroaesthetics Lab/AestheticStability/Stim/shirtCombs.csv")
        self.FacesIdx=pd.read_csv(r"/content/drive/My Drive/GU Neuroaesthetics Lab/AestheticStability/Stim/Makeup.csv")
        self.CarsIdx=pd.read_csv(r"/content/drive/My Drive/GU Neuroaesthetics Lab/AestheticStability/Stim/carCombs.csv")
        
        
        #actual list of stimuli used. to get only the list of stimuli
        self.shirts=natsorted(np.unique([self.ShirtsIdx['Lshirt'],self.ShirtsIdx['Rshirt']]))
        self.faces=natsorted(np.unique([self.FacesIdx['Lface'],self.FacesIdx['Rface']]))
        self.cars=natsorted(np.unique([self.CarsIdx['Lcar'],self.CarsIdx['Rcar']]))
        
        #filler matrix for case of missing counts (in case some comparisons weren't made/missed)
        self.shirtFiller=pd.DataFrame(np.repeat(0,16),index=self.shirts)
        self.facesFiller=pd.DataFrame(np.repeat(0,16),index=self.faces)
        self.carsFiller=pd.DataFrame(np.repeat(0,16),index=self.cars)
        
        # declare the comparison matrices PRE and POST. "comparison matrices": a 1/0 indicator of which stimuli
        # was picked. This format allows easier export into matlab/R, and makes certain calculations easier.
        
        self.pre_FacesComparisonMat_df=pd.DataFrame(columns=self.faces,index=self.faces)
        self.post_FacesComparisonMat_df=pd.DataFrame(columns=self.faces,index=self.faces)
        
        self.pre_CarsComparisonMat_df=pd.DataFrame(columns=self.cars,index=self.cars)
        self.post_CarsComparisonMat_df=pd.DataFrame(columns=self.cars,index=self.cars)
        
        self.pre_ShirtsComparisonMat_df=pd.DataFrame(columns=self.shirts,index=self.shirts)
        self.post_ShirtsComparisonMat_df=pd.DataFrame(columns=self.shirts,index=self.shirts)



        
        """
        building the dataframes:

        here taking data from pavlovia/prolific and parsing out responses associated w/different parts of the experiment
        """
        
        #pre shirts - what was presented, what was picked 'winner', as well as reaction time
        self.preShirts=self.df[['Lshirt','Rshirt']].dropna().reset_index(drop=True)[0:120]
        self.preShirts=pd.concat([self.preShirts,self.df[['preSHIRTSkey_resp.keys','preSHIRTSkey_resp.rt']].dropna().reset_index(drop=True)],axis=1)
        self.preShirts['Winner']=np.where(self.preShirts['preSHIRTSkey_resp.keys']=='left',self.preShirts['Lshirt'],self.preShirts['Rshirt'])
        self.preShirts['Loser']=np.where(self.preShirts['preSHIRTSkey_resp.keys']=='left',self.preShirts['Rshirt'],self.preShirts['Lshirt'])

        #post shirts
        self.postShirts=self.df[['Lshirt','Rshirt']].dropna().reset_index(drop=True)[120::].reset_index(drop=True)
        self.postShirts=pd.concat([self.postShirts,self.df[['postSHIRTSkey_resp.keys','postSHIRTSkey_resp.rt']].dropna().reset_index(drop=True)],axis=1)
        self.postShirts['Winner']=np.where(self.postShirts['postSHIRTSkey_resp.keys']=='left',self.postShirts['Lshirt'],self.postShirts['Rshirt'])
        self.postShirts['Loser']=np.where(self.postShirts['postSHIRTSkey_resp.keys']=='left',self.postShirts['Rshirt'],self.postShirts['Lshirt'])

        
        #pre faces
        self.preFaces=self.df[['Lface','Rface']].dropna().reset_index(drop=True)[0:120]
        self.preFaces=pd.concat([self.preFaces,self.df[['preFACESkey_resp.keys','preFACESkey_resp.rt']].dropna().reset_index(drop=True)],axis=1)
        self.preFaces['Winner']=np.where(self.preFaces['preFACESkey_resp.keys']=='left',self.preFaces['Lface'],self.preFaces['Rface'])
        self.preFaces['Loser']=np.where(self.preFaces['preFACESkey_resp.keys']=='left',self.preFaces['Rface'],self.preFaces['Lface'])

        #post faces
        self.postFaces=self.df[['Lface','Rface']].dropna().reset_index(drop=True)[120::].reset_index(drop=True)
        self.postFaces=pd.concat([self.postFaces,self.df[['postFACESkey_resp.keys','postFACESkey_resp.rt']].dropna().reset_index(drop=True)],axis=1)
        self.postFaces['Winner']=np.where(self.postFaces['postFACESkey_resp.keys']=='left',self.postFaces['Lface'],self.postFaces['Rface'])
        self.postFaces['Loser']=np.where(self.postFaces['postFACESkey_resp.keys']=='left',self.postFaces['Rface'],self.postFaces['Lface'])
        
        #pre cars
        self.preCars=self.df[['Lcar','Rcar']].dropna().reset_index(drop=True)[0:120]
        self.preCars=pd.concat([self.preCars,self.df[['preCARSkey_resp.keys','preCARSkey_resp.rt']].dropna().reset_index(drop=True)],axis=1)
        self.preCars['Winner']=np.where(self.preCars['preCARSkey_resp.keys']=='left',self.preCars['Lcar'],self.preCars['Rcar'])
        self.preCars['Loser']=np.where(self.preCars['preCARSkey_resp.keys']=='left',self.preCars['Rcar'],self.preCars['Lcar'])

        #post cars
        self.postCars=self.df[['Lcar','Rcar']].dropna().reset_index(drop=True)[120::].reset_index(drop=True)
        self.postCars=pd.concat([self.postCars,self.df[['postCARSkey_resp.keys','postCARSkey_resp.rt']].dropna().reset_index(drop=True)],axis=1)
        self.postCars['Winner']=np.where(self.postCars['postCARSkey_resp.keys']=='left',self.postCars['Lcar'],self.postCars['Rcar'])
        self.postCars['Loser']=np.where(self.postCars['postCARSkey_resp.keys']=='left',self.postCars['Rcar'],self.postCars['Lcar'])

        

            
        
        """
        getting the counts for each category, i.e. how many times picked
        """
        
        #counts
        # pre shirts
        self.preShirtsCounts=self.preShirts.Winner.value_counts()#count frequency #series
        self.preShirtsNames=[i for i in self.preShirtsCounts.index]#just the names
        self.preShirtsNames.sort(key=natural_keys)#sort the names in alphabetical then numerical
        self.preShirtsCountsSorted=self.preShirtsCounts[self.preShirtsNames]#get sorted list
        self.preShirtsCountsSorted=pd.DataFrame(self.preShirtsCountsSorted)

        #pre faces
        self.preFacesCounts=self.preFaces.Winner.value_counts()
        self.preFacesNames=[i for i in self.preFacesCounts.index]
        self.preFacesNames.sort(key=natural_keys)
        self.preFacesCountsSorted=self.preFacesCounts[self.preFacesNames]
        self.preFacesCountsSorted=pd.DataFrame(self.preFacesCountsSorted)

        #pre cars
        self.preCarsCounts=self.preCars.Winner.value_counts()
        self.preCarsNames=[i for i in self.preCarsCounts.index]
        self.preCarsNames.sort(key=natural_keys)
        self.preCarsCountsSorted=self.preCarsCounts[self.preCarsNames]
        self.preCarsCountsSorted=pd.DataFrame(self.preCarsCountsSorted)

        #post counts
        
        #post shirts
        self.postShirtsCounts=self.postShirts.Winner.value_counts()
        self.postShirtsNames=[i for i in self.postShirtsCounts.index]
        self.postShirtsNames.sort(key=natural_keys)
        self.postShirtsCountsSorted=self.postShirtsCounts[self.postShirtsNames]
        self.postShirtsCountsSorted=pd.DataFrame(self.postShirtsCountsSorted)
        
        #post faces
        self.postFacesCounts=self.postFaces.Winner.value_counts()
        self.postFacesNames=[i for i in self.postFacesCounts.index]
        self.postFacesNames.sort(key=natural_keys)
        self.postFacesCountsSorted=self.postFacesCounts[self.postFacesNames]
        self.postFacesCountsSorted=pd.DataFrame(self.postFacesCountsSorted)


        #post cars
        self.postCarsCounts=self.postCars.Winner.value_counts()
        self.postCarsNames=[i for i in self.postCarsCounts.index]
        self.postCarsNames.sort(key=natural_keys)
        self.postCarsCountsSorted=self.postCarsCounts[self.postCarsNames]
        self.postCarsCountsSorted=pd.DataFrame(self.postCarsCountsSorted)


        """
        fill in the pre and post comparison matrices
        """

                
        def score(df,stim,winner):
            """ function for comparison matrix in order to give a score 1 or 0, convention is that score
            represents column over row"""
            if winner==stim[0]: #if first of pair is winner
                if np.isnan(df[stim[0]][stim[1]]):# if isnan means first comparison
                    df[stim[0]][stim[1]]=1 #go to that coordinate and put 1 
                    df[stim[1]][stim[0]]=0 #go to symmetric coordinate and make 0
                else:
                    df[stim[0]][stim[1]]+=1#if not nan, means not first comparison, need to add instead of replace
                    df[stim[1]][stim[0]]+=0 #go to symmetric coordinate and add 0
            if winner==stim[1]:
                if np.isnan(df[stim[1]][stim[0]]):
                    df[stim[1]][stim[0]]=1
                    df[stim[0]][stim[1]]=0
                else:
                    df[stim[1]][stim[0]]+=1
                    df[stim[0]][stim[1]]+=0
        
        #faces
        for i in range(len(self.preFaces)):
            score(self.pre_FacesComparisonMat_df,[self.preFaces['Lface'][i],self.preFaces['Rface'][i]],self.preFaces['Winner'][i])
        for i in range(len(self.postFaces)):
            score(self.post_FacesComparisonMat_df,[self.postFaces['Lface'][i],self.postFaces['Rface'][i]],self.postFaces['Winner'][i])
        #shirts
        for i in range(len(self.preShirts)):
            score(self.pre_ShirtsComparisonMat_df,[self.preShirts['Lshirt'][i],self.preShirts['Rshirt'][i]],self.preShirts['Winner'][i])
        for i in range(len(self.postShirts)):
            score(self.post_ShirtsComparisonMat_df,[self.postShirts['Lshirt'][i],self.postShirts['Rshirt'][i]],self.postShirts['Winner'][i])
        #cars
        for i in range(len(self.preCars)):
            score(self.pre_CarsComparisonMat_df,[self.preCars['Lcar'][i],self.preCars['Rcar'][i]],self.preCars['Winner'][i])
        for i in range(len(self.postCars)):
            score(self.post_CarsComparisonMat_df,[self.postCars['Lcar'][i],self.postCars['Rcar'][i]],self.postCars['Winner'][i])


        def fixMissing(self):
            """
            below is a series of loops that look for values in the counts, and if nothing is there, set the count as zero.
            In the case that a stimuli was never picked, its count should be 0, but pavlovia leaves cell empty, so need to do it manually
            """
            for i in self.shirtFiller.index:
                try:#go through
                    self.temp=self.preShirtsCountsSorted.loc[i][0]
                except:#if can't, means empty
                    print('preShirts there was a missing value at '+str(i))
                    self.temp=0#set as zero
                self.shirtFiller.loc[i]=self.temp#make zero in df
            self.preShirtsCountsSorted=self.shirtFiller.copy(deep=True)#deep copy w/zero

            for i in self.shirtFiller.index:
                try:
                    self.temp=self.postShirtsCountsSorted.loc[i][0]
                except:
                    print('postShirts there was a missing value at '+str(i))
                    self.temp=0
                self.shirtFiller.loc[i]=self.temp
            self.postShirtsCountsSorted=self.shirtFiller.copy(deep=True)
            
            for i in self.facesFiller.index:
                try:
                    self.temp=self.preFacesCountsSorted.loc[i][0]
                except:
                    print('preFaces there was a missing value at '+str(i))
                    self.temp=0
                self.facesFiller.loc[i]=self.temp
            self.preFacesCountsSorted=self.facesFiller.copy(deep=True)

            for i in self.facesFiller.index:
                try:
                    self.temp=self.postFacesCountsSorted.loc[i][0]
                except:
                    print('postFaces there was a missing value at '+str(i))
                    self.temp=0
                self.facesFiller.loc[i]=self.temp
            self.postFacesCountsSorted=self.facesFiller.copy(deep=True)
            
            for i in self.carsFiller.index:
                try:
                    self.temp=self.preCarsCountsSorted.loc[i][0]
                except:
                    print('preCars there was a missing value at '+str(i))
                    self.temp=0
                self.carsFiller.loc[i]=self.temp
            self.preCarsCountsSorted=self.carsFiller.copy(deep=True)

            for i in self.carsFiller.index:
                try:
                    self.temp=self.postCarsCountsSorted.loc[i][0]
                except:
                    print('postCars there was a missing value at '+str(i))
                    self.temp=0
                self.carsFiller.loc[i]=self.temp
            self.postCarsCountsSorted=self.carsFiller.copy(deep=True)
    
        """
        call function!
        """
        fixMissing(self)
        
        ## now put counts DF's together
        self.ShirtsCounts=pd.concat([self.preShirtsCountsSorted,self.postShirtsCountsSorted],axis=1)
        self.ShirtsCounts.columns=[self.subname+'PRE',self.subname+'POST']
        self.ShirtsCounts['RankIdx']=np.arange(1,17)#numerical index because idx right now is stimuli
        
        self.FacesCounts=pd.concat([self.preFacesCountsSorted,self.postFacesCountsSorted],axis=1)
        self.FacesCounts.columns=[self.subname+'PRE',self.subname+'POST']
        self.FacesCounts['RankIdx']=np.arange(1,17)

        self.CarsCounts=pd.concat([self.preCarsCountsSorted,self.postCarsCountsSorted],axis=1)
        self.CarsCounts.columns=[self.subname+'PRE',self.subname+'POST']
        self.CarsCounts['RankIdx']=np.arange(1,17)
        
        # add diff column
        #difference in counts from pre to post, calculated as post-pre, negative indicates pre was bigger 
        self.CarsCounts['DIFF']=np.round(self.CarsCounts[self.subname+'POST']-self.CarsCounts[self.subname+'PRE'])
        self.ShirtsCounts['DIFF']=np.round(self.ShirtsCounts[self.subname+'POST']-self.ShirtsCounts[self.subname+'PRE'])
        self.FacesCounts['DIFF']=np.round(self.FacesCounts[self.subname+'POST']-self.FacesCounts[self.subname+'PRE'])
        
        #add sign column-sign of difference. negative indicates pre was bigger
        self.CarsCounts['Sign']=[math.copysign(1,i) for i in self.CarsCounts.DIFF]
        self.ShirtsCounts['Sign']=[math.copysign(1,i) for i in self.ShirtsCounts.DIFF]
        self.FacesCounts['Sign']=[math.copysign(1,i) for i in self.FacesCounts.DIFF]


        ##some useful lists 

        #descending rank by count
        self.preShirtsRanked=list(self.ShirtsCounts.sort_values(by=self.subname+'PRE',ascending=False).index.values)
        self.postShirtsRanked=list(self.ShirtsCounts.sort_values(by=self.subname+'POST',ascending=False).index.values)
        
        self.preFacesRanked=list(self.FacesCounts.sort_values(by=self.subname+'PRE',ascending=False).index.values)
        self.postFacesRanked=list(self.FacesCounts.sort_values(by=self.subname+'POST',ascending=False).index.values)
        
        self.preCarsRanked=list(self.CarsCounts.sort_values(by=self.subname+'PRE',ascending=False).index.values)
        self.postCarsRanked=list(self.CarsCounts.sort_values(by=self.subname+'POST',ascending=False).index.values)
        
        # get rank IDX
        self.preShirtsRankedIDX=(self.ShirtsCounts.sort_values(by=self.subname+'PRE',ascending=False)['RankIdx'].values)
        self.postShirtsRankedIDX=(self.ShirtsCounts.sort_values(by=self.subname+'POST',ascending=False)['RankIdx'].values)
        
        self.preFacesRankedIDX=(self.FacesCounts.sort_values(by=self.subname+'PRE',ascending=False)['RankIdx'].values)
        self.postFacesRankedIDX=(self.FacesCounts.sort_values(by=self.subname+'POST',ascending=False)['RankIdx'].values)
        
        self.preCarsRankedIDX=(self.CarsCounts.sort_values(by=self.subname+'PRE',ascending=False)['RankIdx'].values)
        self.postCarsRankedIDX=(self.CarsCounts.sort_values(by=self.subname+'POST',ascending=False)['RankIdx'].values)
        
    
#%% Functions to run on subject class


"""
BT functions- these functions carry out Bradley Terry tests for the competitions
"""

def MaxLikBT(M,n,npar):
    Wi=np.sum(M).values
    par=np.ones([1,npar])/npar
    for i in range(1000):
        pi = np.ones([npar,1])*par;
        pj = pi.T
        par = Wi/np.sum(n/(pi+pj),axis=1)
        par = par/np.sum(par)
    return(par[::-1])
def BTTest(M,par,npar,n):
    M1=np.ones([npar,1])*par;#creates a 12x12 matrix from par, so now 12 rows of par which was 1x12
    M2=M1.T
    pij = M2/(M1+M2)
    np.fill_diagonal(pij,0)
    return(pij)
def getPij(compMat,n,npar):
    """
    the wrapper function, gives back matrix of competition probabilities
    """
    par=MaxLikBT(compMat,n,npar)
    pij=BTTest(compMat.to_numpy(),par,npar,n)
    return(pij)






"""
FIX: if an item never wins, return divide by zero error.
"""

"""
start count and BT functions
"""
# subname.Shirts_dfprobs=getPij(subname.pre_ShirtsComparisonMat_df+subname.post_ShirtsComparisonMat_df,2,16)#n and npar
# subname.Shirts_dfprobs=pd.DataFrame(subname.Shirts_dfprobs)
# subname.Shirts_dfprobs.index=subname.shirts
# subname.Shirts_dfprobs.columns=subname.shirts

# subname.Cars_dfprobs=getPij(subname.pre_CarsComparisonMat_df+subname.post_CarsComparisonMat_df,2,16)#n and npar
# subname.Cars_dfprobs=pd.DataFrame(subname.Cars_dfprobs)
# subname.Cars_dfprobs.index=subname.cars
# subname.Cars_dfprobs.columns=subname.cars

# subname.Faces_dfprobs=getPij(subname.pre_FacesComparisonMat_df+subname.post_FacesComparisonMat_df,2,16)#n and npar
# subname.Faces_dfprobs=pd.DataFrame(subname.Faces_dfprobs)
# subname.Faces_dfprobs.index=subname.faces
# subname.Faces_dfprobs.columns=subname.faces


    
#%% plotting functions
def radarPlot(subname,category):
    hues=np.linspace(0,1,16,endpoint=False)
    hues=['%1.2f' % i for i in hues]
    hues=[float(i) for i in hues]
    colors = plt.cm.hsv(hues)
    # Compute pie slices
    N = 16
    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    radii = np.repeat(1,N)
    width = np.repeat(0.25,N)
    
    if category=='cars':
        #cars
        counts_df1=subname.preCarsCountsSorted
        counts_df2=subname.postCarsCountsSorted
        categories1=[str(i) for i in counts_df1.index]#cars
        N = len(categories1)
        #cars
        values1=[i for i in counts_df1[0].values]
        values1 += values1[:1] #makes it circular
        values2=[i for i in counts_df2[0].values]
        values2 += values2[:1] #makes it circular
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        #this doesn't change for all. all have 16 data points
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories1, color='grey', size=8)
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([2,4,6,8,10,12,16],[],color="black", size=5)
        plt.ylim(0,16)
        # Plot data
        ax.plot(angles, values1, linewidth=1, linestyle='solid')
        ax.plot(angles, values2, linewidth=1, linestyle='solid')
        # Fill area
        ax.fill(angles, values1, 'b', alpha=0.1)
        ax.fill(angles, values2, 'r', alpha=0.1)
        bars = ax.bar(theta, radii, width=width, bottom=15, color=colors)
        ax.legend(['Pre','Post'])
        plt.title(subname.subname+' '+'Cars')

    if category=='shirts':
        #shirts
        counts_df1=subname.preShirtsCountsSorted
        counts_df2=subname.postShirtsCountsSorted
        categories1=[str(i) for i in counts_df1.index]#shirts
        N = len(categories1)
        #shirts
        values1=[i for i in counts_df1[0].values]
        values1 += values1[:1] #makes it circular
        values2=[i for i in counts_df2[0].values]
        values2 += values2[:1] #makes it circular
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        #this doesn't change for all. all have 16 data points
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories1, color='grey', size=8)
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([2,4,6,8,10,12,16],[],color="black", size=5)
        plt.ylim(0,16)
        # Plot data
        ax.plot(angles, values1, linewidth=1, linestyle='solid')
        ax.plot(angles, values2, linewidth=1, linestyle='solid')
        # Fill area
        ax.fill(angles, values1, 'b', alpha=0.1)
        ax.fill(angles, values2, 'r', alpha=0.1)
        bars = ax.bar(theta, radii, width=width, bottom=15, color=colors)
        ax.legend(['Pre','Post'])
        plt.title(subname.subname+' '+'Shirts')
        plt.rcParams["figure.figsize"] = (10,10)        


    if category=='faces':
        #faces
        counts_df1=subname.preFacesCountsSorted
        counts_df2=subname.postFacesCountsSorted
        categories1=[str(i) for i in counts_df1.index]#faces
        N = len(categories1)
        #faces
        values1=[i for i in counts_df1[0].values]
        values1 += values1[:1] #makes it circular
        values2=[i for i in counts_df2[0].values]
        values2 += values2[:1] #makes it circular
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        #this doesn't change for all. all have 16 data points
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        # Initialise the spider plot
        ax = plt.subplot(111, polar=True)
        # Draw one axe per variable + add labels labels yet
        plt.xticks(angles[:-1], categories1, color='grey', size=8)
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([2,4,6,8,10,12,16],[],color="black", size=5)
        plt.ylim(0,16)
        # Plot data
        ax.plot(angles, values1, linewidth=1, linestyle='solid')
        ax.plot(angles, values2, linewidth=1, linestyle='solid')
        # Fill area
        ax.fill(angles, values1, 'b', alpha=0.1)
        ax.fill(angles, values2, 'r', alpha=0.1)
        #bars = ax.bar(theta, radii, width=width, bottom=15, color=colors)
        ax.legend(['Pre','Post'])
        plt.title(subname.subname+' '+'Faces')
#%% personality test scoring functions
def getBIS(subname):
  """
  http://www.impulsivity.org/measurement/bis11
  gives back a dictionary with BIS scores 
  subname should be a subject object eg. subs['sub1']
  """
  subname.BIS=subname.df[['BISslider.response','BISslider.rt','Question']].dropna().reset_index(drop=True)#get rows w relevant responses
  BISIdx=list(np.arange(1,31))#establish an index
  missingFlag=False#in case missing questions
  if len(subname.BIS)!=30:#if missing
    missingFlag=True#raise flag
    print('has ' +str(30-(len(subname.BIS)))+' missing questions')
    #figure out what is missing, below is a list for reference
    BISqList=['I plan tasks carefully',
    'I do things without thinking',
    'I make-up my mind quickly',
    'I am happy-go-lucky',
    'I don’t “pay attention"',
    'I have “racing” thoughts',
    'I plan trips well ahead of time',
    'I am self controlled',
    'I concentrate easily',
    ' I save regularly',
    'I “squirm” at plays or lectures',
    ' I am a careful thinker.',
    ' I plan for job security.',
    ' I say things without thinking.',
    ' I like to think about complex problems.',
    ' I change jobs.',
    ' I act “on impulse.”',
    'I get easily bored when solving thought problems.',
    'I act on the spur of the moment.',
    ' I am a steady thinker.',
    'I change residences.',
    ' I buy things on impulse.',
    ' I can only think about one thing at a time.',
    ' I change hobbies.',
    ' I spend or charge more than I earn.',
    ' I often have extraneous thoughts when thinking.',
    ' I am more interested in the present than the future.',
    ' I am restless at the theater or lectures.',
    'I like puzzles.',
    ' I am future oriented.']

    missing=list(set(BISqList)-set(subname.BIS['Question']))#what is missing from our question column
    missingIDX=[]
    for i in missing:
      """
      go through and see the original index of each of the missing questions
      """
      missingIDX.append(BISqList.index(i))#this index is 0 onwards!
    missingIDX=[i+1 for i in missingIDX]#start from 1
    print(missingIDX)

  #indices for certain attributes, norm means normal scoring, rev is reversed
  #scoring is 1-4
  AttentionIDX=[5,6,9,11,20,24,26,28]
  MotorIDX=[2,3,4,16,17,19,21,22,23,25,30]
  NonPlanningIDX=[1,7,8,10,12,13,14,15,18,27,29]
  AttentionNorm=[5,6,11,24,26,28]
  AttentionRev=[9,20]
  MotorNorm=[2,3,4,16,17,19,21,22,23,25]
  MotorRev=[30]
  NonPlanningNorm=[14,18,27]
  NonPlanningRev=[1,7,8,10,12,13,15,29]

  def removeItem(item,lst):
    if item in lst:
      lst.remove(item)
  
  if missingFlag:
    """
    if missing questions, then need to remove from
    the trait indices, as well as overall index
    """
    [removeItem(i,AttentionIDX) for i in missingIDX]
    [removeItem(i,MotorIDX) for i in missingIDX]
    [removeItem(i,NonPlanningIDX) for i in missingIDX]
    [removeItem(i,AttentionNorm) for i in missingIDX]
    [removeItem(i,MotorNorm) for i in missingIDX]
    [removeItem(i,NonPlanningNorm) for i in missingIDX]
    [removeItem(i,AttentionRev) for i in missingIDX]
    [removeItem(i,MotorRev) for i in missingIDX]
    [removeItem(i,NonPlanningRev) for i in missingIDX]
    [removeItem(i,BISIdx) for i in missingIDX]

  
  #need to reindex with those missing taken out
  subname.BIS.index=BISIdx


  #get all items associated w/ 3 categores
  AttentionBIS=[subname.BIS.loc[i]['BISslider.response'] for i in AttentionIDX]
  MotorBIS=[subname.BIS.loc[i]['BISslider.response']  for i in MotorIDX]
  NonPlanningBIS=[subname.BIS.loc[i]['BISslider.response']  for i in NonPlanningIDX]
  #get Norm items
  AttentionNormVal=[subname.BIS.loc[i]['BISslider.response'] for i in AttentionNorm]
  MotorNormVal=[subname.BIS.loc[i]['BISslider.response']  for i in MotorNorm]
  NonPlanningNormVal=[subname.BIS.loc[i]['BISslider.response']  for i in NonPlanningNorm]
  #norm scores
  AttentionNormScore=np.sum(AttentionNormVal)
  MotorNormScore=np.sum(MotorNormVal)
  NonPlanningNormScore=np.sum(NonPlanningNormVal)
  #get all items associated w/reversed
  AttentionRevVal=[subname.BIS.loc[i]['BISslider.response'] for i in AttentionRev]
  MotorRevVal=[subname.BIS.loc[i]['BISslider.response']  for i in MotorRev]
  NonPlanningRevVal=[subname.BIS.loc[i]['BISslider.response']  for i in NonPlanningRev]
  #reverse scores
  reverseBISdict={1:4,2:3,3:2,4:1} #reversed score dict  
  AttentionRevScore=np.sum([reverseBISdict[i] for i in AttentionRevVal])
  MotorRevScore=np.sum([reverseBISdict[i] for i in MotorRevVal])
  NonPlanningRevScore=np.sum([reverseBISdict[i] for i in NonPlanningRevVal])
  #calculate category specific scores
  AttentionBIS_SCORE=AttentionNormScore+AttentionRevScore
  MotorBIS_SCORE=MotorNormScore+MotorRevScore
  NonPlanningBIS_SCORE=NonPlanningNormScore+NonPlanningRevScore
  OverallBIS_SCORE=AttentionBIS_SCORE+MotorBIS_SCORE+NonPlanningBIS_SCORE
  #BIS dictionary
  subname.Big5Dict={'Attention':AttentionBIS_SCORE,'Motor':MotorBIS_SCORE,'NonPlanning':NonPlanningBIS_SCORE,'Overall':OverallBIS_SCORE}

def getBIG5(subname):

  """
  https://openpsychometrics.org/printable/big-five-personality-test.pdf
  gives back a dictionary with Big5 scores 
  subname should be a subject object eg. subs['sub1']
  """
  subname.Big5=subname.df[['BIG5slider.response','BIG5slider.rt','items']].dropna().reset_index(drop=True)#get rows w/big5 responses, items is questions
  Big5Idx=list(np.arange(1,51))#establish a 1+index
  missingFlag=False#flag for missing questions
  if len(subname.Big5)!=50:#means something is missing
    missingFlag=True#raise flag
    print('has ' +str(50-(len(subname.Big5)))+' missing questions')
    #figure out what is missing
    #list of questions for reference
    Big5Questions=['Am the life of the party.',
      'Feel little concern for others.',
      'Am always prepared.',
      'Get stressed out easily.',
      'Have a rich vocabulary.',
      "Don't talk a lot.",
      'Am interested in people.',
      'Leave my belongings around.',
      'Am relaxed most of the time.',
      'Have difficulty understanding abstract ideas.',
      'Feel comfortable around people.',
      'Insult people.',
      'Pay attention to details.',
      'Worry about things.',
      'Have a vivid imagination.',
      'Keep in the background.',
      "Sympathize with other's feelings.",
      'Make a mess of things.',
      'Seldom feel blue.',
      'Am not interested in abstract ideas.',
      'Start conversations.',
      "Am not interested in other people's problems.",
      'Get chores done right away.',
      'Am easily disturbed.',
      'Have excellent ideas.',
      'Have little to say.',
      'Have a soft heart.',
      'Often forget to put things back in their proper place.',
      'Get upset easily.',
      'Do not have a good imagination.',
      'Talk to a lot of different people at parties.',
      'Am not really interested in others.',
      'Like order.',
      'Change my mood a lot.',
      'Am quick to understand things.',
      "Don't like to draw attention to myself.",
      'Take time out for others.',
      'Shirk(Avoid) my duties.',
      'Have frequent mood swings.',
      'Use difficult words.',
      "Don't mind being the center of attention.",
      "Feel other's emotions.",
      'Follow a schedule.',
      'Get irritated easily.',
      'Spend time reflecting on things.',
      'Am quiet around strangers.',
      'Make people feel at ease.',
      'Am exacting in my work.',
      'Often feel blue.',
      'Am full of ideas.']

    missing=list(set(Big5Questions)-set(subname.Big5['items']))#what is not in our subject
    missingIDX=[]
    for i in missing:
      """
      go through and see the original index of each of the missing questions
      """
      missingIDX.append(Big5Questions.index(i))#this index is 0 onwards!
    
    missingIDX=[i+1 for i in missingIDX]#start index from 1
    print(missingIDX)
  
  #make certain scores negative (per BIG5 scoring guide)
  minus=[2,4,6,8,10,12,14,16,18,20,22,24,26,28,29,30,32,34,36,38,39,44,46,49]
  
  #indices for Extroversion, Agreeableness.. etc
  E_Idx=[1, 6, 11, 16, 21, 26, 31, 36, 41, 46]
  A_Idx=[2, 7, 12, 17, 22, 27, 32, 37, 42, 47]
  C_Idx=[3, 8, 13, 18, 23, 28, 33, 38, 43, 48]
  N_Idx=[4, 9, 14, 19, 24, 29, 34, 39, 44, 49]
  O_Idx=[5, 10, 15, 20, 30, 35, 40, 45, 50]

  def removeItem(item,lst):
    #if item in list, remove
    if item in lst:
      lst.remove(item)
  
  if missingFlag:
    """
    if missing questions, then need to remove from
    the trait indices, as well as overall index
    """
    [removeItem(i,E_Idx) for i in missingIDX]
    [removeItem(i,A_Idx) for i in missingIDX]
    [removeItem(i,C_Idx) for i in missingIDX]
    [removeItem(i,N_Idx) for i in missingIDX]
    [removeItem(i,O_Idx) for i in missingIDX]
    [removeItem(i,minus) for i in missingIDX]
    [removeItem(i,Big5Idx) for i in missingIDX]
  
  #need to reindex with those missing taken out
  subname.Big5.index=Big5Idx
  
  #make certain scores negative
  subname.Big5.loc[minus,'BIG5slider.response']=subname.Big5.loc[minus,'BIG5slider.response']*-1
  
  #get the response values for each trait
  E_Vals=[subname.Big5.loc[i]['BIG5slider.response'] for i in E_Idx]
  A_Vals=[subname.Big5.loc[i]['BIG5slider.response'] for i in A_Idx]
  C_Vals=[subname.Big5.loc[i]['BIG5slider.response'] for i in C_Idx]
  N_Vals=[subname.Big5.loc[i]['BIG5slider.response'] for i in N_Idx]
  O_Vals=[subname.Big5.loc[i]['BIG5slider.response'] for i in O_Idx]
  
  #sum the scores, the integers in front represent a scoring convention
  EScore=20+np.sum(E_Vals)
  AScore=14+np.sum(A_Vals)
  CScore=14+np.sum(C_Vals)
  NScore=38+np.sum(N_Vals)
  OScore=8+np.sum(O_Vals)
  
  #return a dictionary which becomes an attribute of the subjecct
  subname.Big5Dict={'Extroversion':EScore,'Agreeableness':AScore,'Conscientiousness':CScore,'Neuroticism':NScore,'Openness':OScore}

