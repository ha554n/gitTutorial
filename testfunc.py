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

        """
        below is a series of loops that look for values in the counts, and if nothing is there, set the count as zero.
        In the case that a stimuli was never picked, its count should be 0, but pavlovia leaves cell empty, so need to do it manually
        """
        def fixMissing(self):
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
        self.CarsCounts['DIFF']=np.round(self.CarsCounts[self.subname+'PRE']-self.CarsCounts[self.subname+'POST'])
        self.ShirtsCounts['DIFF']=np.round(self.ShirtsCounts[self.subname+'PRE']-self.ShirtsCounts[self.subname+'POST'])
        self.FacesCounts['DIFF']=np.round(self.FacesCounts[self.subname+'PRE']-self.FacesCounts[self.subname+'POST'])
        
        #add sign column-sign of difference. negative indicates pre was bigger
        self.CarsCounts['Sign']=[math.copysign(1,i) for i in self.CarsCounts.DIFF]
        self.ShirtsCounts['Sign']=[math.copysign(1,i) for i in self.ShirtsCounts.DIFF]
        self.FacesCounts['Sign']=[math.copysign(1,i) for i in self.FacesCounts.DIFF]

        def equalize(countsdf):
          """
          looks where pre and post are tied (have equal # of winnings for an item)
          and breaks those ties by seeing whether the neighbors align more with pre 
          or with post. 
          does so by looking at values in DIFF, and changes the Sign column accordingly

          n= number of neighbors to look above and below the tie
          """
          tieIdx=np.where(countsdf.DIFF==0)[0]#check where ties
          # print('ties at')
          # print(tieIdx)
          # print('the tied items are ')
          tieStims=[countsdf.index[i] for i in tieIdx]
          # print(tieStims)
          for i,k  in zip(tieIdx,tieStims):
            # print(i,k)
            n=6
            upper_i=[]
            lower_i=[]
            for j in np.arange(1,n):
              upper_i.append(np.arange(len(countsdf))[(i-j)%len(countsdf)])
              lower_i.append(np.arange(len(countsdf))[(i+j)%len(countsdf)])
              # print(['idx='+str(i),'before'+str(upper_i), 'after'+str(lower_i)])
            # print('the upper/lower bound for '+ str(i))
            # print(upper_i)
            # print(lower_i)
            counter=0
            neighbors=0
            while neighbors==0:
              # print('counter is ' +str(counter))
              upper=countsdf.DIFF[upper_i[counter]]
              lower=countsdf.DIFF[lower_i[counter]]
              # print('neighbors are '+str(upper)+ ' and '+ str(lower))
              neighbors=countsdf.DIFF[upper_i[counter]]+countsdf.DIFF[lower_i[counter]]
              # print('neighbors total is ' +str(neighbors))
              counter+=1
            if neighbors>1:
              # print('made positive')
              countsdf.loc[k,'Sign']=1
            elif neighbors<1:
              # print('made negative')
              countsdf.loc[k,'Sign']=-1
        
        equalize(self.CarsCounts)
        equalize(self.ShirtsCounts)
        equalize(self.FacesCounts)


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
        
        self.Shirts_dfprobs=getPij(self.pre_ShirtsComparisonMat_df+self.post_ShirtsComparisonMat_df,2,16)#n and npar
        self.Shirts_dfprobs=pd.DataFrame(self.Shirts_dfprobs)
        self.Shirts_dfprobs.index=self.shirts
        self.Shirts_dfprobs.columns=self.shirts
        
        self.Cars_dfprobs=getPij(self.pre_CarsComparisonMat_df+self.post_CarsComparisonMat_df,2,16)#n and npar
        self.Cars_dfprobs=pd.DataFrame(self.Cars_dfprobs)
        self.Cars_dfprobs.index=self.cars
        self.Cars_dfprobs.columns=self.cars
        
        self.Faces_dfprobs=getPij(self.pre_FacesComparisonMat_df+self.post_FacesComparisonMat_df,2,16)#n and npar
        self.Faces_dfprobs=pd.DataFrame(self.Faces_dfprobs)
        self.Faces_dfprobs.index=self.faces
        self.Faces_dfprobs.columns=self.faces

        self.ShirtsStims=[i for i in combinations(self.shirts,2)]
        self.CarsStims=[i for i in combinations(self.cars,2)]
        self.FacesStims=[i for i in combinations(self.faces,2)]


        
        """
        get the subjects own counts
        """

        def getPreLengths(competitiondf,npar):
            '''
            gets the indices where 'pre' is greater than 'post',npar is the number of items,competition df is a df of counts
            pre and post
            '''
            pre=np.where(competitiondf[self.subname+'PRE']>=competitiondf[self.subname+'POST'])[0]#indices where pre>post
            addCircle=all(x in pre for x in [npar-1,0])#if any of the strings are at edges then have to make circular
            prelist=[]
            for k,g in groupby(enumerate(pre),lambda ix:ix[0]-ix[1]):
                #this returns consecutive numbers. ex. if strings are [2,3,4,6,7,8,11,12]
                #will return [[2,3,4],[6,7,8],[11,12]]
                prelist.append(list(map(itemgetter(1),g)))
            prelens=[len(i) for i in prelist]#get length of each string
            if addCircle:#if strings at edges
                #fix lengths
                prelens[-1]=prelens[0]+prelens[-1]
                prelens.pop(0)
                #fix lists indices
                lnf=[[item for sublist in [prelist[-1],prelist[0]] for item in sublist]]#join the last and first items
                rest=prelist[1:-1]#get everyting except last and first
                joined=[rest,lnf]# put together
                prelist=[item for sublist in joined for item in sublist]
            return([prelist,prelens])
        def getPostLengths(competitiondf,npar):
            post=np.where(competitiondf[self.subname+'PRE']<=competitiondf[self.subname+'POST'])[0]
            addCircle=all(x in post for x in [npar-1,0])
            postlist=[]
            for k,g in groupby(enumerate(post),lambda ix:ix[0]-ix[1]):
                postlist.append(list(map(itemgetter(1),g)))
                #     print('string indices')
                #     print(postlist)
            postlens=[len(i) for i in postlist]
            if addCircle:
                postlens[-1]=postlens[0]+postlens[-1]
                postlens.pop(0)
                #fix lists indices
                lnf=[[item for sublist in [postlist[-1],postlist[0]] for item in sublist]]#join the last and first items
                rest=postlist[1:-1]#get everyting except last and first
                joined=[rest,lnf]# put together
                postlist=[item for sublist in joined for item in sublist]
                  #     print('string lengths')
            return([postlist,postlens])
        def getTrueCountsInfo(countsdf,npar):
            
            """
            combines count getter and does further computations on them.
            """
            [preIdx,preLen]=getPreLengths(countsdf,npar)#get pre strings
            [postIdx,postLen]=getPostLengths(countsdf,npar)#get post strings
            preSum=np.abs([np.sum(countsdf['DIFF'][i]) for i in preIdx])#sum of each string pre
            postSum=np.abs([np.sum(countsdf['DIFF'][i]) for i in postIdx]) #sum of each string post
            preSum=[i for i in preSum]
            postSum=[i for i in postSum]
            allSum=np.concatenate([preSum,postSum])#combined sum of all strings
            
            maxPreDiff=np.max(preSum)#what is the max difference in the pre string
            maxPostDiff=np.max(postSum)
            maxAllDiff=np.max(allSum)#max difference overall
            
            preNum=len(preLen)#how many strings are there in pre
            postNum=len(postLen)
            allNum=preNum+postNum

            avePreLen=np.sum(preLen)/preNum#what is the average length of a string in pre
            avePostLen=np.sum(postLen)/postNum
            aveAllLen=(avePreLen+avePostLen)/2

            totPreSum=np.sum(preSum)#overall sum of all strings in pre (sum of DIFF column)
            totPostSum=np.sum(postSum)
            totAllSum=np.sum([totPreSum,totPostSum])

            avePreMag=np.sum(preSum)/preNum#average sum of each string
            avePostMag=np.sum(postSum)/postNum
            aveAllMag=(avePreMag+avePostMag)/2

            medPreMag=np.median(preSum)#median sum of each string
            medPostMag=np.median(postSum)
            medAllMag=np.median([preSum+postSum])

            outDF=pd.DataFrame([preNum,postNum,allNum,avePreLen,avePostLen,aveAllLen,totPreSum,totPostSum,totAllSum,maxPreDiff,maxPostDiff,maxAllDiff,avePreMag,avePostMag,aveAllMag,medPreMag,medPostMag,medAllMag]).T
            outDF.columns=['PreStr','PostStr', 'AllStr','avePreLen','avePostLen','aveAllLen','totPreSum','totPostSum','totAllSum','maxPreDiff','maxPostDiff','maxAllDiff','avePreMag','avePostMag','aveAllMag','medPreMag','medPostMag','medAllMag']

            return(outDF)
          
        
        self.Cars_trueCountsInfo=getTrueCountsInfo(self.CarsCounts,16)#16 is the number of items
        self.Shirts_trueCountsInfo=getTrueCountsInfo(self.ShirtsCounts,16)
        self.Faces_trueCountsInfo=getTrueCountsInfo(self.FacesCounts,16)

    def runBTsim(self,itera):

        def compete_ha(df,pij,stim,dfprobs):
            """

            runs a competition based on a probability matrix
            in this case pWin means prob of column beating row

            df=comparisonMatrix (should be empty/nans)
            pij= matrix of competition probs
            stim= df of competitions

            """

            pWin=dfprobs[stim[0]][stim[1]]
            result=np.random.choice([0,1],p=[pWin,1-pWin])
            winner=stim[result]

            if winner==stim[0]: #if first of pair is winner
                if np.isnan(df[stim[0]][stim[1]]):#if is nan
                    df[stim[0]][stim[1]]=1 #go to that coordinate and put 1 
                else:
                    df[stim[0]][stim[1]]+=1#if not nan, then can add to it

                if np.isnan(df[stim[1]][stim[0]]): #if symmetric pair is nan
                    df[stim[1]][stim[0]]=0 #go to symmetric coordinate and set to 0
                else:
                    None
                    #keep it to where it is
            if winner==stim[1]: #if first of pair is winner
                if np.isnan(df[stim[1]][stim[0]]):#if is nan
                    df[stim[1]][stim[0]]=1 #go to that coordinate and put 1 
                else:
                    df[stim[1]][stim[0]]+=1#if not nan, then can add to it

                if np.isnan(df[stim[0]][stim[1]]): #if symmetric pair is nan
                    df[stim[0]][stim[1]]=0 #go to symmetric coordinate and set to 0
                else:
                    None
        def getPreLengths(competitiondf,npar):
            pre=np.where(competitiondf.PRE>=competitiondf.POST)[0]
            addCircle=all(x in pre for x in [npar-1,0])#subtracting one because of index
            prelist=[]
            for k,g in groupby(enumerate(pre),lambda ix:ix[0]-ix[1]):
                prelist.append(list(map(itemgetter(1),g)))
            prelens=[len(i) for i in prelist]
            if addCircle:
                #fix lengths
                prelens[-1]=prelens[0]+prelens[-1]
                prelens.pop(0)
                #fix lists indices
                lnf=[[item for sublist in [prelist[-1],prelist[0]] for item in sublist]]#join the last and first items
                rest=prelist[1:-1]#get everyting except last and first
                joined=[rest,lnf]# put together
                prelist=[item for sublist in joined for item in sublist]
            return([prelist,prelens])
        def getPostLengths(competitiondf,npar):
            post=np.where(competitiondf.PRE<=competitiondf.POST)[0]
            addCircle=all(x in post for x in [npar-1,0])#subtracting one because of index
            postlist=[]
            for k,g in groupby(enumerate(post),lambda ix:ix[0]-ix[1]):
                postlist.append(list(map(itemgetter(1),g)))
        #     print('string indices')
        #     print(postlist)
            postlens=[len(i) for i in postlist]
            if addCircle:
                postlens[-1]=postlens[0]+postlens[-1]
                postlens.pop(0)
                #fix lists indices
                lnf=[[item for sublist in [postlist[-1],postlist[0]] for item in sublist]]#join the last and first items
                rest=postlist[1:-1]#get everyting except last and first
                joined=[rest,lnf]# put together
                postlist=[item for sublist in joined for item in sublist]
        #     print('string lengths')
            return([postlist,postlens])
          
        def equalize(countsdf):
          """
          looks where pre and post are tied (have equal # of winnings for an item)
          and breaks those ties by seeing whether the neighbors align more with pre 
          or with post. 
          does so by looking at values in DIFF, and changes the Sign column accordingly

          n= number of neighbors to look above and below the tie
          """
          tieIdx=np.where(countsdf.DIFF==0)[0]#check where ties
          # print('ties at')
          # print(tieIdx)
          # print('the tied items are ')
          tieStims=[countsdf.index[i] for i in tieIdx]
          # print(tieStims)
          for i,k  in zip(tieIdx,tieStims):
            # print(i,k)
            n=6
            upper_i=[]
            lower_i=[]
            for j in np.arange(1,n):
              upper_i.append(np.arange(len(countsdf))[(i-j)%len(countsdf)])
              lower_i.append(np.arange(len(countsdf))[(i+j)%len(countsdf)])
              # print(['idx='+str(i),'before'+str(upper_i), 'after'+str(lower_i)])
            # print('the upper/lower bound for '+ str(i))
            # print(upper_i)
            # print(lower_i)
            counter=0
            neighbors=0
            while neighbors==0:
              # print('counter is ' +str(counter))
              upper=countsdf.DIFF[upper_i[counter]]
              lower=countsdf.DIFF[lower_i[counter]]
              # print('neighbors are '+str(upper)+ ' and '+ str(lower))
              neighbors=countsdf.DIFF[upper_i[counter]]+countsdf.DIFF[lower_i[counter]]
              # print('neighbors total is ' +str(neighbors))
              counter+=1
            if neighbors>1:
              # print('made positive')
              countsdf.loc[k,'Sign']=1
            elif neighbors<1:
              # print('made negative')
              countsdf.loc[k,'Sign']=-1

        def getCountsInfo(countsdf,npar):
            #combines counts getters as well as adds other things like magnitude and sum
            """
            combines count getter and does further computations on them.
            """
            [preIdx,preLen]=getPreLengths(countsdf,npar)#get pre strings
            [postIdx,postLen]=getPostLengths(countsdf,npar)#get post strings
            preSum=np.abs([np.sum(countsdf['DIFF'][i]) for i in preIdx])#sum of each string pre
            postSum=np.abs([np.sum(countsdf['DIFF'][i]) for i in postIdx]) #sum of each string post
            preSum=[i for i in preSum]
            postSum=[i for i in postSum]
            allSum=np.concatenate([preSum,postSum])#combined sum of all strings
            
            maxPreDiff=np.max(preSum)#what is the max difference in the pre string
            maxPostDiff=np.max(postSum)
            maxAllDiff=np.max(allSum)#max difference overall
            
            preNum=len(preLen)#how many strings are there in pre
            postNum=len(postLen)
            allNum=preNum+postNum

            avePreLen=np.sum(preLen)/preNum#what is the average length of a string in pre
            avePostLen=np.sum(postLen)/postNum
            aveAllLen=(avePreLen+avePostLen)/2

            totPreSum=np.sum(preSum)#overall sum of all strings in pre (sum of DIFF column)
            totPostSum=np.sum(postSum)
            totAllSum=np.sum([totPreSum,totPostSum])

            avePreMag=np.sum(preSum)/preNum#average sum of each string
            avePostMag=np.sum(postSum)/postNum
            aveAllMag=(avePreMag+avePostMag)/2

            medPreMag=np.median(preSum)#median sum of each string
            medPostMag=np.median(postSum)
            medAllMag=np.median([preSum+postSum])

            outDF=pd.DataFrame([preNum,postNum,allNum,avePreLen,avePostLen,aveAllLen,totPreSum,totPostSum,totAllSum,maxPreDiff,maxPostDiff,maxAllDiff,avePreMag,avePostMag,aveAllMag,medPreMag,medPostMag,medAllMag]).T
            outDF.columns=['PreStr','PostStr', 'AllStr','avePreLen','avePostLen','aveAllLen','totPreSum','totPostSum','totAllSum','maxPreDiff','maxPostDiff','maxAllDiff','avePreMag','avePostMag','aveAllMag','medPreMag','medPostMag','medAllMag']

            return(outDF)

        def simulateBT_ha(idx,stims,npar,dfprobs):
            #pre and post runs

            pre_CompetitionMat_df=pd.DataFrame(columns=idx,index=idx)
            post_CompetitionMat_df=pd.DataFrame(columns=idx,index=idx)
            for i in stims:
                """
                fill in pre and post competition matrix
                """
                compete_ha(pre_CompetitionMat_df,dfprobs,[i[0],i[1]],dfprobs)
                compete_ha(post_CompetitionMat_df,dfprobs,[i[0],i[1]],dfprobs)

            AllCompetition=pd.DataFrame(np.sum(pre_CompetitionMat_df)+np.random.random(len(idx)),columns=['PRE'])#add random noise to break ties
            AllCompetition['POST']=np.sum(post_CompetitionMat_df)+np.random.random(len(idx))#add random noise to break ties
            AllCompetition['DIFF']=np.round(AllCompetition.PRE-AllCompetition.POST)
            AllCompetition['Sign']=[math.copysign(1,i) for i in AllCompetition.DIFF]
            
            equalize(AllCompetition)
            countsInfo=getCountsInfo(AllCompetition,npar)
            return(countsInfo)
        
        """
        shirts simulation
        """

        self.Shirts_sim=pd.DataFrame()
        start=time.process_time()
        
        for i in range(itera):
            self.Shirts_sim=self.Shirts_sim.append(simulateBT_ha(self.shirts,self.ShirtsStims,16,self.Shirts_dfprobs))
        end=time.process_time()
        print(str(end-start) +' s')
        
        

        """
        faces simulation
        """
        
        self.Faces_sim=pd.DataFrame()
        start=time.process_time()
        
        for i in range(itera):
            self.Faces_sim=self.Faces_sim.append(simulateBT_ha(self.faces,self.FacesStims,16,self.Faces_dfprobs))
        end=time.process_time()
        print(str(end-start) +' s')
      
    """
    start plotting functions
    """

    def radarPlot(self,category):
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
            counts_df1=self.preCarsCountsSorted
            counts_df2=self.postCarsCountsSorted
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
            plt.title(self.subname+' '+'Cars')

        if category=='shirts':
            #shirts
            counts_df1=self.preShirtsCountsSorted
            counts_df2=self.postShirtsCountsSorted
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
            plt.title(self.subname+' '+'Shirts')
            plt.rcParams["figure.figsize"] = (10,10)        


        if category=='faces':
            #faces
            counts_df1=self.preFacesCountsSorted
            counts_df2=self.postFacesCountsSorted
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
            plt.title(self.subname+' '+'Faces')

      