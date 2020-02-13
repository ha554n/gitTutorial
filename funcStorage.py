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

def getPreLengths(subname,countsdf,npar,kind):
    '''
    gets the indices where 'pre' is greater than 'post',npar is the number of items,competition df is a df of counts
    pre and post
    '''
    if kind=='sign':
      pre=np.where(countsdf.Sign<0)[0]
    elif kind=='diff':
      pre=np.where(countsdf[subname+'PRE']>=countsdf[subname+'POST'])[0]#indices where pre=>post
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
def getPostLengths(subname,countsdf,npar,kind):

    if kind=='sign':
      post=np.where(countsdf.Sign>0)[0]
    elif kind=='diff':
      post=np.where(countsdf[subname+'POST']>=countsdf[subname+'PRE'])[0]
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