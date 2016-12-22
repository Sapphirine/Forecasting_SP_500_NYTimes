lib <- c('zoo','tm','plyr','textir','lubridate','glmnet','ranger')
sapply(lib, require,character.only = TRUE)
clean <- function(rawstr){
    rawstr <- gsub("<.*?>"," ",rawstr)
    rawstr <- gsub("\n", " ",rawstr)
    tmobj <- VectorSource(rawstr)
    tmcor <- Corpus(tmobj)
    tmcor <- tm_map(tmcor, content_transformer(tolower))
    tmcor <- tm_map(tmcor, removePunctuation)
    tmcor <- tm_map(tmcor, removeNumbers)
    tmcor <- tm_map(tmcor, removeWords, c(stopwords("english")))
    tmcor <- tm_map(tmcor, stripWhitespace)
    matp <- DocumentTermMatrix(tmcor)
    return (as.matrix(matp))
}
tokenize <- function(i, j, c, dm, FUN=clean){
  if (i==j-1){
      return (rbind.fill.matrix(FUN(dm[i,c]),FUN(dm[j,c])))
  }
  else if(i==j){
      return (FUN(dm[i,c]))
  }
  else{
    t1 <- tokenize(i,(j-i)%/%2+i,c, dm, FUN)
    t2 <- tokenize((j-i)%/%2+i+1,j, c,dm,FUN)
    return (rbind.fill.matrix(t1,t2))
  }
}
getshare <- function(i,j,s,c,dm){
  inds <- sample(1:nrow(dm),nrow(dm),replace = FALSE)
  dm <- dm[inds,]
  if (i==j){
    sharedm <- tokenize(i*s+1,min((i+1)*s,nrow(dm)),c,dm)
    sharename <- colnames(sharedm)
    return (list(sharedm, sharename))
  }
  else if(i==j-1){
    dm1 <- tokenize(i*s+1,min((i+1)*s,nrow(dm)),c,dm)
    name1 <- colnames(dm1)
    dm2 <- tokenize(j*s+1,min((j+1)*s,nrow(dm)),c,dm)
    name2 <- colnames(dm2)
    sharename <- intersect(name1,name2)
    sharedm <- rbind(dm1[,sharename],dm2[,sharename])
    return (list(sharedm, sharename))
  }
  else{
    r1 <- getshare(i,(j-i)%/%2+i, s, c, dm)
    r2 <- getshare((j-i)%/%2+i+1,j,s, c, dm)
    sharename <- intersect(r1[[2]],r2[[2]])
    sharedm <- rbind(r1[[1]][,sharename],r2[[1]][,sharename])
    return (list(sharedm, sharename))
  }
}
Feature_Gen <- function(data,st,en){
  pdlp <- getshare(st,en,ceiling(nrow(data)/(en-st+1)),3,data)[[1]] #need to be set to control the size of data
  pdlp[is.na(pdlp)] <- 0
  pdh <-  getshare(st,en,ceiling(nrow(data)/(en-st+1)),2,data)[[1]] #same as aforementioned.
  pdh[is.na(pdh)] <- 0
  pdlp <- tfidf(pdlp)
  pdh <- tfidf(pdh)
  pdlp <- pdlp[,apply(pdlp, 2, var) !=0]
  pdh <- pdh[,apply(pdh, 2, var) !=0]
  colnames(pdlp) <- paste('lead_para',colnames(pdlp),sep="_")
  colnames(pdh) <- paste('head',colnames(pdh),sep="_")
  df <- as.data.frame(cbind(pdlp,pdh))
  df$lwcs <- data$lwcs
  df$pcs <- data$pcs
  df$DATE <- data$DATE
  return (df)
}
##We can generate tf-idf for 1950s in the following way
setwd('~/Desktop/1950-1990/1950-1990')
sp <- read.csv('~/Desktop/SP500_data.csv',stringsAsFactors=FALSE)
spr <- data.frame(ret=diff(log(sp$adj.close)))
spr$ret <- ifelse(spr$ret>0,1,0)
spr$DATE <- as.Date(sp$date[1:(nrow(sp)-1)],'%m/%d/%Y')
###read all data in 1950
pd <- process_data('19501_all.csv')
for (i in 2:12){
  dname <- paste(paste('1950',as.character(i),sep=""),'_all.csv',sep="")
  pd <- rbind(pd,process_data(dname))
}
for (j in 1951:1959){
  for (i in 1:12){
    dname <- paste(paste(as.character(j),as.character(i),sep=""),'_all.csv',sep="")
    pd <- rbind(pd,process_data(dname))
  }
}
###tfidf matrix
out <- Feature_Gen(pd)
pd_1950s <- pd
out_1950s <- out
setwd('~/Desktop/1950-1990')
write.csv(out,"df_1950s.csv")
####The process is the same for other decades