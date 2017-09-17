---
title: "Facebook Recruiting IV"
category : kaggle
tagline: "A lesson in overfitting"
tags : [kaggle, R, RandomForest]
---

The ['Facebook Recruiting IV: Human or bot?'](https://www.kaggle.com/c/facebook-recruiting-iv-human-or-bot) competition just ended on Kaggle. For those unfamiliar with the competition, participants downloaded a table of about 7 million bids, which corresponded to another table of around 6,000 bidders. For 4,000 of those bidders, you had to estimate the probability that they were a human or bot, based on the remaining 2,000 bidders, whose bot status was given.

I was first on the public leaderboard for some time, but ended up coming in 17th on the private leaderboard. Still, given I'm a beginner I was pretty happy with the outcome, and I learned a lot. In this post I'll give an overview of my approach to feature engineering and modelling, and share some of the lessons learned. Everything was done in R.

## Feature extraction

Each subheading below describes a general group of features that I extracted from the data and used in modelling. Features were estimated for each bidder_id in the training and test sets, and combined into a matrix called ‘bidderCharacteristics’.

### Reading in the data

{% highlight r %}
bids<-fread("Downloaded/bids.csv", sep = ",", header= TRUE)
train<-fread("Downloaded/train.csv", sep = ",", header= TRUE)
test<-fread("Downloaded/test.csv", sep = ",", header= TRUE)
{% endhighlight %}

### Total number of unique bids, countries, devices, IPs, URLs, merch categories for each bidder
The first set of features were just simple sums of the number of unique(x) where x is one of the variables in the bids table. The below code shows how I calculated the number of unique countries each bidder bid from.

{% highlight r %}
#How many countries for each bidder?
nCountries<-data.frame(numCountries=with(bids, tapply(country, bidder_id, FUN = function(x) length(unique(x)))))
bidderCharacteristics<-merge(cbind(nCountries,bidder_id=row.names(nCountries)),bidderCharacteristics,all.y=T)
{% endhighlight %}

### Proportion of bids in each country, device, IP, URL, merchandise category
These features proved to be useful, particularly country. The below code shows how this was calculated for the country feature – a similar process was used for the other variables.

{% highlight r %}
#Country proportions?
bidderIDsByCountry<-round(with(bids,table(bidder_id,country)),0)
bidderIDsByCountry<-as.matrix(bidderIDsByCountry/rowSums(bidderIDsByCountry))
bidderCharacteristics<-data.frame(bidder_id=unique(bids$bidder_id))
bidderCharacteristics<-data.frame(bidder_id=bidderCharacteristics,cty=matrix(0,length(bidderCharacteristics),ncol(bidderIDsByCountry)))
bidderCharacteristics<-bidderCharacteristics[order(bidderCharacteristics$bidder_id),]
bidderCharacteristics[,2:(ncol(bidderIDsByCountry)+1)]<-as.numeric(as.matrix(bidderIDsByCountry[,1:ncol(bidderIDsByCountry)]))
colnames(bidderCharacteristics)[2:length(colnames(bidderCharacteristics))]<-c('cty.none',paste('cty.',colnames(bidderIDsByCountry)[2:length(colnames(bidderIDsByCountry))],sep=""))
{% endhighlight %}

Given the sheer number of IPs and URLs, I limited the lists of these to only those IPs and URLs that were used for at least 1000 bids. This still ended up giving me about 600 IP variables and 300 URL variables. Correlation plots showed highly-correlated clusters of some URLs and IPs. To reduce dimensionality I tried using principal components analysis (PCA) on these variables. It seemed to help with URLs; in my final model I included the top 50 principal components from the URLs. Not much help was provided with IPs – I ended up using RandomForest importance scores on the full set to decide on which ones to include, which wasn’t many in the end. Here’s the code used to perform pca on the urls:

{% highlight r %}
urls<-bidderCharacteristics[,grep("url\\.",colnames(bidderCharacteristics))]
urls<-removeLowVarianceCols(urls,4)
url.pca<-prcomp(urls, scale.=TRUE)
urls<-predict(url.pca,urls)[,1:50]
bidderCharacteristics<-cbind(bidderCharacteristics,url=urls)
{% endhighlight %}

### Mean popularity of country, device, IP, URL, merchandise categories used
I defined the ‘popularity’ of a particular country, device, or etc., as the number of unique bidder_id’s that bid from that variable. For each bidder, I then took the mean of these popularity scores given the countries, devices, etc., that they bid from. Here's the code snippet used to calculate mean IP popularity:
{% highlight r %}
#Mean popularity of IPs used
nBidderIDsPerIP<-data.frame(numBidderIDsPerIP=with(bids, tapply(bidder_id, ip, FUN = function(x) length(unique(x)))))
ipPopularity<-subset(bids[!duplicated(subset(bids, select=c(bidder_id,ip)))],select=c(bidder_id,ip))
ipPopularity<-merge(ipPopularity,cbind(as.data.table(nBidderIDsPerIP),ip=row.names(nBidderIDsPerIP)),by="ip",all.x=T)
ipPopularity<-data.frame(ipPop=with(ipPopularity, tapply(numBidderIDsPerIP, bidder_id, FUN = function(x) mean(x))))
rm(nBidderIDsPerIP)
bidderCharacteristics<-merge(cbind(ipPopularity,bidder_id=row.names(ipPopularity)),bidderCharacteristics,all.y=T)
{% endhighlight %}
### Mean number of bids from countries, devices, IPs, etc., bidded from
Very similar to the previous feature: this looked at how many bids were made from each country, device, etc., and then gave each bidder the mean of these features across the countries, devices, etc., that the bidded from.
{% highlight r %}
#Mean number of bids for Countrys bidded from
nBidsPerCountry<-data.frame(numBidsEachCountry=with(bids, tapply(bid_id, country, FUN = function(x) length(x))))
biddersAndCountrys<-subset(bids[!duplicated(subset(bids, select=c(bidder_id,country)))],select=c(bidder_id,country))
biddersAndCountrys<-merge(cbind(country=row.names(nBidsPerCountry),nBidsPerCountry),biddersAndCountrys,by.x='country',by.y='country',all.x=T)
biddersAndCountrys<-data.frame(meanCountryPopularity=with(biddersAndCountrys, tapply(numBidsEachCountry, bidder_id, FUN = function(x) mean(x))))
rm(nBidsPerCountry)
bidderCharacteristics<-merge(cbind(biddersAndCountrys,bidder_id=row.names(biddersAndCountrys)),bidderCharacteristics,all.y=T)
{% endhighlight %}
### Time domain
As it appears many others in this competition also realised, it became clear fairly early on to me in the competition that the bids were from three distinct three-day time periods, and that the time between the first and last bid was probably very close to exactly 31 days. Based on this information I could convert the obfuscated ‘time’ field into more meaningful units. 
{% highlight r %}
day<-(max(bids$time)-min(bids$time))/31
hour<-day/24
bids$hour24<-floor(bids$time/hour)%%24
{% endhighlight %}
A number of other features stemmed from having this info on hand…

### Proportion of bids in each hour of the day
A plot of bot density by hour of the day showed bots were more common during the ‘off-peak’ bidding periods. This suggested that taking the total proportion of a user’s bids in each hour of the day was likely to be a useful feature. The below code shows how I did this:
{% highlight r %}
bidsPerTimeSlotPerBidder<-data.frame(with(bids, tapply(bid_id, list(bidder_id,hour24), FUN = function(x) length(x))))
bidsPerTimeSlotPerBidder[is.na(bidsPerTimeSlotPerBidder)]<-0
bidsPerTimeSlotPerBidder<-bidsPerTimeSlotPerBidder/rowSums(bidsPerTimeSlotPerBidder)
bidderCharacteristics<-merge(cbind(bidsPerTimeSlotPerBidder,bidder_id=row.names(bidsPerTimeSlotPerBidder)),bidderCharacteristics,all.y=T)
{% endhighlight %}
### ‘Bids per time’, mean time between bids, and ‘time active’
Bids per time and mean time between bids are self-explanatory. Time active I defined as the time between a bidder’s first and last bid.

I originally extracted these three features using the entire bids table at once. I later realised however, that the features could be skewed by the fact that there were three distinct time chunks. For instance, the mean time between a user’s bids was calculated by looking at the average length of time between a user’s bids. If a user had two bids in separate time chunks, this metric would be artificially inflated by the missing data between the time chunks.

Thus I figured out the ‘cut off times’ for each bid chunk’s start and end, divided into three chunks, extracted my features from each, and then took the overall means of the three features:
{% highlight r %}
#This section calculates the 'bid response time', how long between bids, and the total time active (in each 3 day 'chunk')
bidsO<-bids[order(auction,time)]
#Reduce unnecessary granularity of the time field
bidsO$time<-floor(bidsO$time/1e6)
cutoffTime1<-9670569*1e9/1e6
cutoffTime2<-9734233*1e9/1e6

#Split the bids into three chunks according to the cut off times
bidsTimeChunk<-list(
  time1<-bidsO[which(bidsO$time<=cutoffTime1),]
  ,time2<-bidsO[which(bidsO$time>cutoffTime1&bidsO$time<cutoffTime2),]
  ,time3<-bidsO[which(bidsO$time>=cutoffTime2),]
)

#Initialisation
meanTimeDiffByBidder<-list()
firstLastBid<-list()
numBids<-list()
overallMean<-0

#Calculate mean difference in time between bids for each bidder.
#Do this by lagging the bids table by one bid, then subtracting the lagged bid time from the original. Then take the average of this for each bidder.
for (i in 1:3){
  bidsTimeChunk[[i]]$auctionL1<-c(bidsTimeChunk[[i]]$auction[1],bidsTimeChunk[[i]]$auction[1:(nrow(bidsTimeChunk[[i]])-1)])
  bidsTimeChunk[[i]]$timeDiff<-bidsTimeChunk[[i]]$time-c(bidsTimeChunk[[i]]$time[1],bidsTimeChunk[[i]]$time[1:(nrow(bidsTimeChunk[[i]])-1)])
  bidsTimeChunk[[i]]$timeDiff[1]<-NA
  bidsTimeChunk[[i]]$timeDiff[which(bidsTimeChunk[[i]]$auction!=bidsTimeChunk[[i]]$auctionL1)]<-NA
  meanTimeDiffByBidder[[i]]<-ddply(bidsTimeChunk[[i]],~bidder_id,summarise,mean=mean(timeDiff,na.rm=T))
  overallMean<-overallMean+mean(meanTimeDiffByBidder[[i]][,2],na.rm=T)*nrow(bidsTimeChunk[[i]])/nrow(bidsO)
}
#Replace any NAs with the overall mean
for (i in 1:3){
  meanTimeDiffByBidder[[i]][which(is.na(meanTimeDiffByBidder[[i]][,2])),2]<-overallMean
}

#Calculated ‘bids per time’ and ‘time active’
bidsPerTime<-list()
timeActive<-list()
overallMean<-0
for (i in 1:3){
  firstLastBid[[i]]<-ddply(bidsTimeChunk[[i]],~bidder_id,summarise,firstBid=min(time,na.rm=T),lastBid=max(time,na.rm=T))
  firstLastBid[[i]]$timeActive<-firstLastBid[[i]]$lastBid-firstLastBid[[i]]$firstBid
  numBids[[i]]<-data.frame(numBids=with(bidsTimeChunk[[i]], tapply(bid_id, bidder_id, FUN = function(x) length(x))))
  firstLastBid[[i]]$bidsPerTime<-ifelse(numBids[[i]]$numBids>1,numBids[[i]]$numBids/firstLastBid[[i]]$timeActive,NA)
  firstLastBid[[i]]$bidsPerTime[which(firstLastBid[[i]]$bidsPerTime==Inf)]<-NA
  overallMean<-overallMean+mean(firstLastBid[[i]]$bidsPerTime,na.rm=T)*nrow(bidsTimeChunk[[i]])/nrow(bidsO)
}
for (i in 1:3){
  firstLastBid[[i]]$bidsPerTime[which(is.na(firstLastBid[[i]]$bidsPerTime))]<-overallMean
  bidsPerTime[[i]]<-subset(firstLastBid[[i]],select=c(bidder_id,bidsPerTime))
  timeActive[[i]]<-subset(firstLastBid[[i]],select=c(bidder_id,timeActive))
}

#Take the average 'bid response time' for each bidder over the three time chunks
meanTimeDiffByBidder<-merge(merge(meanTimeDiffByBidder[[1]],meanTimeDiffByBidder[[2]],by.x='bidder_id',by.y='bidder_id',all.x=T,all.y=T)
                            ,meanTimeDiffByBidder[[3]],by.x='bidder_id',by.y='bidder_id',all.x=T,all.y=T)
meanTimeDiffByBidder<-data.frame(bidder_id=meanTimeDiffByBidder[,1],meanTimeBwBids=rowMeans(meanTimeDiffByBidder[,2:4],na.rm=T))
#Take the average of nBids/(last bid - first bid) for each bidder over the three time chunks
bidsPerTime<-merge(merge(bidsPerTime[[1]],bidsPerTime[[2]],by.x='bidder_id',by.y='bidder_id',all.x=T,all.y=T)
                   ,bidsPerTime[[3]],by.x='bidder_id',by.y='bidder_id',all.x=T,all.y=T)
bidsPerTime<-data.frame(bidder_id=bidsPerTime[,1],bidsPerTime=rowMeans(bidsPerTime[,2:4],na.rm=T))
#Take the sum of (last bid - first bid) for each bidder over the three time chunks
timeActive<-merge(merge(timeActive[[1]],timeActive[[2]],by.x='bidder_id',by.y='bidder_id',all.x=T,all.y=T)
                  ,timeActive[[3]],by.x='bidder_id',by.y='bidder_id',all.x=T,all.y=T)
timeActive<-data.frame(bidder_id=timeActive[,1],timeActive=rowSums(timeActive[,2:4],na.rm=T))
#Add to bidder characteristics matrix
bidderCharacteristics<-merge(meanTimeDiffByBidder,bidderCharacteristics,by.x='bidder_id',by.y='bidder_id',all.y=T)
bidderCharacteristics<-merge(bidsPerTime,bidderCharacteristics,by.x='bidder_id',by.y='bidder_id',all.y=T)
bidderCharacteristics<-merge(timeActive,bidderCharacteristics,by.x='bidder_id',by.y='bidder_id',all.y=T)
rm('bidsPerHour','meanBidsPerHour','varInBidsPerHour','countriesPerHour','meanCountriesPerHour','varInCountriesPerHour','auctionsPerHour','meanAuctionsPerHour','varInAuctionsPerHour','devicesPerHour','meanDevicesPerHour','varInDevicesPerHour','firstLastBid','numBids','overallMean','bidsTimeChunk','meanTimeDiffByBidder','time1','time2','time3','bidsO','bidderIDsByCountry')
{% endhighlight %}
### Proportion of auctions where a bidder was the last bidder
I took this feature as potentially meaning the bidder won the auction.
{% highlight r %}
#Propn of auctions where they were the final bidder..
lastBidsOnAuction<-ddply(bids,~auction,summarise,time=max(time,na.rm=T))
lastBidsOnAuction <- merge(lastBidsOnAuction, bids, by.x=c("auction","time"), by.y=c("auction","time"))
nLastBids<-data.frame(numLastBids=with(lastBidsOnAuction, tapply(bid_id, bidder_id, FUN = function(x) length(x))))
bidderCharacteristics<-merge(cbind(nLastBids,bidder_id=row.names(nLastBids)),bidderCharacteristics,all.y=T)
bidderCharacteristics$numLastBids[which(is.na(bidderCharacteristics$numLastBids))]<-0
bidderCharacteristics$finalBidRate<-bidderCharacteristics$numLastBids/bidderCharacteristics$numAuctions
{% endhighlight %}
### Mean duration of auctions a bidder participated in
This didn’t turn out to be particularly useful:

{% highlight r %}
#Mean duration of auctions participated in
auctionDurations<-ddply(bids,~auction,summarise,firstBid=min(time/1e6,na.rm=T),lastBid=max(time/1e6,na.rm=T))
auctionDurations$dur<-auctionDurations$lastBid-auctionDurations$firstBid
auctionDurations[,2:4]<-auctionDurations[,2:4]/(hour/1e6)
auctionDurations[,2:3]<-auctionDurations[,2:3]-min(auctionDurations[,2])
auctionDurations<-data.frame(with(cbind(bids,dur=auctionDurations$dur[match(bids$auction,auctionDurations$auction)]), tapply(dur, list(bidder_id), FUN = function(x) mean(x))))  
bidderCharacteristics$auctionDurations<-auctionDurations[match(bidderCharacteristics$bidder_id,rownames(auctionDurations)),1]
rm(auctionDurations)
{% endhighlight %}

### Variance in proportion of bids in each hour
The idea here was that a human might be more varied in terms of the hours of the day the bidded in, or maybe the opposite. For each of the 9 days I calculated the 
{% highlight r %}
#Variance in proportion of bids in each hour...
bids$hour<-floor(bids$time/hour)
bids$hour24<-floor(bids$time/hour)%%24
bids$day<-floor(bids$time/day)
bids$hour<-bids$hour-min(bids$hour)
bids$day<-bids$day-min(bids$day)
bidsInEachHour<-data.frame(with(bids, tapply(bid_id, list(bidder_id,day,hour24), FUN = function(x) length(x))))
for (d in unique(bids$day)){
    bidsInEachHour[,grep(paste("X",d,"\\.",sep=''),colnames(bidsInEachHour))]<-
      bidsInEachHour[,grep(paste("X",d,"\\.",sep=''),colnames(bidsInEachHour))]/
      rowSums(bidsInEachHour[,grep(paste("X",d,"\\.",sep=''),colnames(bidsInEachHour))],na.rm=T)
}
bidsInEachHour[is.na(bidsInEachHour)]<-0
propnBids<-list()
varPropnBids<-list()
for (n in 0:23){
  propnBids[[n+1]]<-as.data.frame(bidsInEachHour[,grep(paste("\\.",n,sep=''),colnames(bidsInEachHour))],
                                  bidder_id=row.names(bidsInEachHour))
  propnBids[[n+1]]<-apply(propnBids[[n+1]],1,function (x) var(x,na.rm=T))  
  bidderCharacteristics<-cbind(bidderCharacteristics,propnBids[[n+1]][
    match(names(propnBids[[n+1]]),bidderCharacteristics$bidder_id)
    ])
}
colnames(bidderCharacteristics)[(ncol(bidderCharacteristics)-23):ncol(bidderCharacteristics)]<-paste("hrVar",0:23,sep="")
{% endhighlight %}
### Mean, variance, skewness and kurtosis of bids per auction, bids per device… auctions per device, auctions per county… and so on
Using the example of auctions per country, this feature was extracted by creating a table of bidder_id’s by countries and then placing the number of unique auctions in each country/bidder_id combination in the table cells. Row-wise mean, variance, skewness and kurtosis were then obtained. This was repeated for many possible combination of IPs, URLs, bids, auctions, devices, countries and hours:
{% highlight r %}
meanVarSkewKurt<-function(inData){
  mean<-apply(inData, 1, mean, na.rm=T)
  var<-apply(inData, 1, sd, na.rm=T)
  mean[is.na(mean)]<-mean(mean,na.rm=T)
  var<-var/mean
  var[is.na(var)]<-mean(var,na.rm=T)
  skewness<-apply(inData, 1, skewness, na.rm=T)
  kurtosis<-apply(inData, 1, kurtosis, na.rm=T)
  skewness<-skewness/mean
  kurtosis<-kurtosis/mean
  skewness[is.na(skewness)]<-mean(skewness,na.rm=T)
  kurtosis[is.na(kurtosis)]<-mean(kurtosis,na.rm=T)
  if (sum(names(mean)==names(skewness))==6614&(sum(names(mean)==names(var))==6614)&(sum(names(mean)==names(kurtosis))==6614)) {
    return(data.frame(row.names=names(mean),mean,var,skewness,kurtosis))
  } else {
    return("ERR")
  }
}
bids$hour<-floor(bids$time/hour)
names<-list()
big<-data.frame(row.names=unique(bids$bidder_id)[order(unique(bids$bidder_id))])
system.time({
for (xPer in c('ip','url','bid_id','auction','device','country')){
  for (yy in c('auction','device','hour','country')){
    if (xPer != yy){
      print(paste(gsub("_id","",xPer),"sPer",yy,sep=""))
      big<-data.frame(big,
                      meanVarSkewKurt(data.frame(with(bids, tapply(get(xPer), list(bidder_id,get(yy)), FUN = function(x) length(unique(x))))))
      )
      if (ncol(big)==4){
        colnames(big)<-paste(gsub("_id","",xPer),"sPer",yy,".",c('m','v','s','k'),sep="")
      } else {
        colnames(big)<-c(colnames(big)[1:(length(colnames(big))-4)],paste(gsub("_id","",xPer),"sPer",yy,".",c('m','v','s','k'),sep=""))  
      }
    }
  }
}
})
bidderCharacteristics<-merge(bidderCharacteristics,big,by.x='bidder_id',by.y='row.names',all.x=T)
{% endhighlight %}
### Clean up
{% highlight r %}
rm(list=ls(all=T)[!(ls(all=T)%in%c('bidderCharacteristics','oversampleOnes','removeLowVarianceCols','removeZeroVarianceCols','wtc','test','train','.Random.seed'))])
{% endhighlight %}


## Modelling

### Choice of model

To set a benchmark, I first tried modelling the bot probability using logistic regression. As expected this wasn't particularly effective. RandomForest was the next model I tried. I also experimented with adaboost and extraTrees from the caret package, as well as xgboost. None of these were able to outperform RandomForest, however.

### Addressing class imbalance

In the training set of some ~2000 bidders, there were only about 100 bots. I found I was able to improve both local cross validation (CV) and public leaderboard scores by over-sampling the bots prior to training the model. I achieved this through an R function:

{% highlight r %}
oversampleOnes<-function(dataIn,m){
  out<-dataIn
  for (z in 1:m){
    out<-rbind(out,dataIn[dataIn$outcome==1,])
  }
  return(out)
}
{% endhighlight %}

### Local training and testing - cross validation

While I did experiment with the cross validation features packaged with caret, I preferred the flexibilty of my own CV routine. I used 5- or 10-fold CV, depending on how much time I wanted to wait for resutls (usually used 10-fold).

I found my public leaderboard scores were usually higher than my CV scores, which I thought was a bit strange. I was probably overfitting the public leaderboard to some extent, or just getting lucky, because my final score on the private leaderboard ended up been much closer to my general CV performance.

The below loop gives the general gist of how I trained, tested and tuned my RF model using the training set:


{% highlight r %}

#Create a list of 'evals' to store the evaluation and parameters
if(!exists('i')){evals=list();i=0}
#Use all 8 cpu cores
cores=8
num.chunk=8
#os sets how many times to oversample the bots. os = 8 seemed to give best performance - this meant the entire training set went from having 100 to 900 bots.
os = 8
total.tree=3200;avg.tree <- ceiling(total.tree/num.chunk)
#Iterations is how many CV repeats to do... usually would just set high and stop the model at some point.
iterations=1000
for (iterAtion in 3:iterations){
  set.seed(iterAtion)
  #Initialise samples for 10-fold cross validation
  cv=10
  trainIdx<-list()
  testIdx<-list()
  tmp<-sample(1:nrow(trainChar))
  for (j in 1:cv){
    testIdx[[j]]<-tmp[round((j-1)*floor(nrow(trainChar)/cv)+1,0):min(round(j*floor(nrow(trainChar)/cv)+1,0),length(tmp))]
    trainIdx[[j]]<-tmp[which(!tmp%in%testIdx[[j]])]
  }
  #Initialise multicore:
  cl <- makeCluster(cores, type = "SOCK");registerDoSNOW(cl)
  #These for loops were used for tuning RF parameters like mtry.
  for (mtry in c(18)){
    for (cvIdx in 1:cv){
      print(cvIdx)
      rf_fit <- foreach(ntree = rep(avg.tree, num.chunk), .combine = combine, 
                        .packages = c("randomForest")) %dopar% {
                          randomForest(x=oversampleOnes(trainChar[trainIdx[[cvIdx]],allVars],os)[,-1], 
                                       y=oversampleOnes(trainChar[trainIdx[[cvIdx]],allVars],os)[,1], 
                                       ntree=ntree, mtry=mtry)                                                                 }
      #Make and store predictions and variable importance vector
      if (cvIdx==1){
        imps<-importance(rf_fit,class=1)
        trainCharPredictions<-subset(trainChar,select=c(bidder_id,outcome))
        trainCharPredictions$prediction.rf<-NA
      } else {
        imp<-importance(rf_fit,class=1)
        imps<-imps+imp[match(rownames(imps),rownames(imp))]
      }
      trainCharPredictions$prediction.rf[testIdx[[cvIdx]]]<-predict(rf_fit, trainChar[testIdx[[cvIdx]],allVars], type = "prob")[,2]
      print(paste("RF performance: ",
                  round(slot(performance(prediction(trainCharPredictions$prediction.rf,trainChar$outcome), "auc"), "y.values")[[1]],3),
                  sep=""))
    }
    imps<-imps[order(imps[,1]),]
    eval<-cbind(
      os,mtry,cv,iterAtion,
      slot(performance(prediction(trainCharPredictions$prediction.rf,trainCharPredictions$outcome), "auc"), "y.values")[[1]],
      total.tree,
      length(allVars)
    )
    print(eval)
    colnames(eval)<-c('os','mtry','cv folds','seed','RFAUC','ntrees','nvars')
    i=i+1
    evals[[i]]=list(eval,imps,paste(allVars,collapse=","))
  }
  stopCluster(cl)
}
{% endhighlight %}

### Fitting the final model

After testing models out via cross validation, the below general code snippet was used to make submissions:

{% highlight r %}

mtry=18
total.tree=8000;avg.tree <- ceiling(total.tree/num.chunk)
os=8
cl <- makeCluster(cores, type = "SOCK");registerDoSNOW(cl)
rf_fit_full <- foreach(ntree = rep(avg.tree, num.chunk), .combine = combine, 
                       .packages = c("randomForest")) %dopar% {
                         randomForest(x=oversampleOnes(trainChar[,allVars],os)[,-1], 
                                      y=oversampleOnes(trainChar[,allVars],os)[,1], 
                                      ntree=ntree, mtry=mtry)
                       } #Change to just trainCharRestricted to use entire training set.
stopCluster(cl)
testChar$prediction <- predict(rf_fit_full, testChar, type = "prob")[,2]
#Give bidders not in the bids dataset the average probability of being a bot.
prob<-sum(train$outcome)/nrow(train)
outPred<-merge(testChar,test,by='bidder_id',all.y=T)
outPred<-outPred[,c('bidder_id','prediction')]
outPred[which(is.na(outPred[,2])),2]<-prob
write.csv(outPred,file='submission.csv',row.names=F)

{% endhighlight %}

### Variable selection for the final model

I didn't end up using all of the variables generated during the feature engineering stage in my final model (there were some 1400 in total), though some of my best-scoring models did include as many as 1200 features. The 'core' model had around 315 predictor variables. These particular 315 came out of various tests using RF importance, balanced with my findings on what seemed to just work. When I added the mean/variance/skewness/kurtosis set of features, performance seemed to degrade, so a number of these features ended up being excluded. I tried to address the high dimensionality problem in various ways - reducing sets of highly-correlated variables, and removing variables with low RF importance scores - however none of these seemed to really improve performance. The takeaway from that for me was that RandomForest seems to be very effective at extracting all of the relevant information from the variables you give it, without being confounded by superfluous or barely-relevant variables. I'm not sure if this is always the case, but it seemed to be so here - removing variables that seemed like they should be useless in a statistical sense usually reduced model accuracy.

If you're curious, here is the vector of 'best' variables that I used in the final model (50 URL principal components variables are all that's excluded from this list):

{% highlight r %}

allVars<-c("outcome","numLastBids","timeActive","bidsPerTime","meanTimeBwBids","bidsPerhour.m","bidsPerhour.v","bidsPerhour.s","bidsPerhour.k","auctionsPerhour.m","auctionsPerhour.v","auctionsPerhour.s","auctionsPerhour.k","urlsPerhour.m","urlsPerhour.v","urlsPerhour.s","urlsPerhour.k","ipsPerhour.m","ipsPerhour.v","ipsPerhour.s","ipsPerhour.k","X0","X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12","X13","X14","X15","X16","X17","X18","X19","X20","X21","X22","X23","numURLs","numDevices","numBids","numAuctions","numIPs","numCountries","ipPop","dvc.1","dvc.2","dvc.3","dvc.4","dvc.5","dvc.6","dvc.7","dvc.8","dvc.9","dvc.10","dvc.11","dvc.12","dvc.13","dvc.14","dvc.15","dvc.16","dvc.17","dvc.18","dvc.19","dvc.20","dvc.21","dvc.22","dvc.23","dvc.24","dvc.25","dvc.26","dvc.27","dvc.28","dvc.29","dvc.30","dvc.31","dvc.32","dvc.33","dvc.34","dvc.35","dvc.36","dvc.37","dvc.38","dvc.39","dvc.41","dvc.42","dvc.43","dvc.45","dvc.46","dvc.47","dvc.49","dvc.50","dvc.51","dvc.52","dvc.53","dvc.56","dvc.57","dvc.58","dvc.59","dvc.60","dvc.61","dvc.62","dvc.63","dvc.64","dvc.65","dvc.67","dvc.69","dvc.70","dvc.71","dvc.72","dvc.73","dvc.74","dvc.75","dvc.76","dvc.77","dvc.78","dvc.79","dvc.80","dvc.81","dvc.82","dvc.83","dvc.84","dvc.85","dvc.86","dvc.87","dvc.88","dvc.90","dvc.91","dvc.92","dvc.93","dvc.94","dvc.95","dvc.96","dvc.98","dvc.99","dvc.100","dvc.101","dvc.102","dvc.104","dvc.105","dvc.107","dvc.108","dvc.109","dvc.110","dvc.111","dvc.112","dvc.113","dvc.114","dvc.116","dvc.117","dvc.118","dvc.119","dvc.120","dvc.122","dvc.123","dvc.124","dvc.125","dvc.126","dvc.127","dvc.128","dvc.129","dvc.130","dvc.131","dvc.132","dvc.133","dvc.134","dvc.135","dvc.137","dvc.138","dvc.139","dvc.140","dvc.141","dvc.142","dvc.143","dvc.144","dvc.146","dvc.147","dvc.148","dvc.150","dvc.153","dvc.154","dvc.155","dvc.157","dvc.159","dvc.162","dvc.163","dvc.164","dvc.166","dvc.168","dvc.169","dvc.170","dvc.171","dvc.173","dvc.174","dvc.175","dvc.176","dvc.177","dvc.179","dvc.181","dvc.182","dvc.183","dvc.184","dvc.185","dvc.186","dvc.187","dvc.189","dvc.190","dvc.191","dvc.192","dvc.194","dvc.195","dvc.196","dvc.197","dvc.198","dvc.199","dvc.200","dvc.201","dvc.202","dvc.203","dvc.204","dvc.205","dvc.206","dvc.207","dvc.208","dvc.209","dvc.210","dvc.211","dvc.212","dvc.213","dvc.214","dvc.215","dvc.216","dvc.217","dvc.219","dvc.220","dvc.221","dvc.222","dvc.224","dvc.225","dvc.226","dvc.227","dvc.228","dvc.229","dvc.230","dvc.231","dvc.232","dvc.233","dvc.234","dvc.235","dvc.236","dvc.237","dvc.238","finalBidRate","cty.ae","cty.ar","cty.au","cty.az","cty.bd","cty.bf","cty.bn","cty.br","cty.ca","cty.ch","cty.cn","cty.de","cty.dj","cty.ec","cty.es","cty.et","cty.eu","cty.fr","cty.gt","cty.id","cty.in","cty.it","cty.ke","cty.lk","cty.lt","cty.lv","cty.mr","cty.mx","cty.my","cty.ng","cty.no","cty.np","cty.ph","cty.pk","cty.qa","cty.ro","cty.rs","cty.ru","cty.sa","cty.sd","cty.sg","cty.th","cty.tr","cty.ua","cty.uk","cty.us","cty.vn","cty.za","sumHrVar","url.150","ip.557","ip.283","ip.549","urlPop","auctionDurations","meanURLPopularity","meanIPPopularity","meanCountryPopularity","meanDevicePopularity","countryPop","auctionPop","devicePop","meanAuctionPopularity","ipsPerdevice.m","auctionsPerdevice.m","auctionsPercountry.m","urlsPerdevice.m")

{% endhighlight %}

## Lessons learned and things to try next time

Here are some of the key things I learned from this competition, or things I might do differently next time:

- The private leaderboard can be misleading; next time I will conduct more rigorous testing using local cross validation assessments rather than 'trusting' the public leaderboard.
- Feature engineering is far more important than model selection or parameter tuning (beyond a certain point). Next time I'll focus more on feature extraction and having a clear structure around my feature extraction/variable selection process.
- Upon looking at some of the better-scoring participants solutions, I think it's easy to see why I came 17th, and not higher. Their features were just a bit more logical/clever in terms of being able to pick out the bots. The structure of their feature extraction was also clearer.
- Oversampling to address class imbalances can improve accuracy (at least in an ROC AUC sense).
- Next time I'll save all my plots as I go so I can include some more pretty pictures in a write-up like this!

### Thanks for reading!