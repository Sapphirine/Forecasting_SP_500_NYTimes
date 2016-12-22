lib <- c('zoo','tm','plyr','textir','lubridate','glmnet','ranger','ggplot2','grid')
sapply(lib, require,character.only = TRUE)
merge_func <- function(out){
  out[,'DATE'] <-  as.Date(out[,'DATE'],'%Y-%m-%d')
  df_all<-join(out,sp,by=c("DATE"),type="inner")
  df_wo_date <- df_all[,-1]
  X <- df_wo_date[,-ncol(df_wo_date)]
  y <- df_all[,'rets']
  return (list(X,y,df_all[,'DATE']))
}
lassfit <- function(X,y){
   lasso_out <- glmnet(X,y,alpha=1)
   return (lasso_out)
}
dsis <- function(lasso_out,X,y){
  pred <- predict(lasso_out,s=lasso_out$lambda,newx=X)
  mse <- sqrt(apply(pred, 2, function(x) mean((x-y)**2)))
  j <- which.min(mse)
  bestn <- lasso_out$df[j]
  f_ind <- which(lasso_out$beta[,j]!=0)
  return (list(pred[,which.min(mse)],min(mse),bestn,f_ind))
}
#Load R data, which is more efficiently stored than .cvs
load("~/Desktop/1950-1990/res/RES_1950S_F.RData")
load("~/Desktop/1950-1990/res/RES_1960S_F.RData")
load("~/Desktop/1950-1990/res/RES_1970S_F.RData")
load("~/Desktop/1950-1990/res/RES_1980S_F.RData")
load("~/Desktop/1950-1990/res/RES_1990S_F.RData")
load("~/Desktop/1950-1990/res/RES_2000S_F.RData")
load("~/Desktop/1950-1990/res/RES_2010S_F.RData")
sp <- read.csv('~/Desktop/SP500_data.csv',stringsAsFactors=FALSE)
sp$y <- sp$adj.close
sp$adj.close <- NULL
sp$DATE <- as.Date(sp$date,'%m/%d/%Y')
sp$date <- NULL
tmp <- sp[2:(nrow(sp)),ncol(sp)]
sp <- data.frame(rets=diff(log(sp[,1])),DATE=tmp)
#Main Loops
louts <- list(0,0,0,0,0,0,0)
pres <- list(0,0,0,0,0,0,0)
mses <- rep(0,7)
bestns <- rep(0,7)
f_inds <- list(0,0,0,0,0,0,0)
ys <- list(0,0,0,0,0,0,0)
dates <- list(0,0,0,0,0,0,0)
outs <- list(out_1950s,out_1960s,out_1970s,out_1980s,out_1990s,out_2000s,out_2010s)
i <- 1
timestart<-Sys.time()
for (out in outs){
  dataxy <- merge_func(out)
  X <- dataxy[[1]]
  y <- dataxy[[2]]
  dates[[i]] <- dataxy[[3]] 
  ys[[i]] <- y
  #write.table(cbind(y,X)[1:10,1:10],paste("data",as.character(i),sep="_"),
   #           sep=",",col.names=FALSE,row.names = FALSE)
  lasso_out <- lassfit(X,y)
  louts[[i]] <- lasso_out
  res <- dsis(lasso_out,X,y)
  pres[[i]] <- res[[1]]
  mses[i] <- res[[2]]
  bestns[i] <- res[[3]]
  f_inds[[i]] <- res[[4]]
  i <- i + 1
}
print (timestart-Sys.time())
#Visualization
TIMES <- c(1950,1960,1970,1980,1990,2000,2010)
df <- as.data.frame(TIMES)
df$Square_Error <- mses
df$n_features <- sapply(outs,ncol)
df$effective_prop <- bestns/df$n_features
p1 <- ggplot(df,aes(TIMES, y =Square_Error)) + 
    geom_line(colour=888) + 
  scale_x_continuous(breaks = c(195:201)*10,
                     labels=c("1950s","1960s","1970s","1980s","1990s","2000s","2010s"))
p2 <- ggplot(df, aes(x = TIMES, y = n_features)) + 
    geom_line(colour=123) + 
  scale_x_continuous(breaks = c(195:201)*10,
                     labels=c("1950s","1960s","1970s","1980s","1990s","2000s","2010s"))
p3 <- ggplot(df, aes(x = TIMES, y = effective_prop)) + 
    geom_line(colour=666) + 
  scale_x_continuous(breaks = c(195:201)*10,
                     labels=c("1950s","1960s","1970s","1980s","1990s","2000s","2010s"))
grid.newpage()
pushViewport(viewport(layout = grid.layout(4, 1, 
                                               heights = unit(c(1,rep(5,3)), 
                                                              "null"))))
grid.text(label = 'Summary_GLMNET', vp = viewport(layout.pos.row = 1, layout.pos.col = 1),
              gp=gpar(col="black", fontsize=12))
print(p1, vp = viewport(layout.pos.row = 2, layout.pos.col = 1))         
print(p2, vp = viewport(layout.pos.row = 3, layout.pos.col = 1))
print(p3, vp = viewport(layout.pos.row = 4, layout.pos.col = 1))

Error_Rate_abs <- list((pres[[1]] - ys[[1]])**2)
for (i in 2:7){
  Error_Rate_abs <- c(Error_Rate_abs,list((pres[[i]] - ys[[i]])**2))
}
dfes <- Error_Rate_abs[[1]]
times <- rep(TIMES[1],length(Error_Rate_abs[[1]]))
for (i in (2:7)){
  dfes <- c(dfes,Error_Rate_abs[[i]])
  times <- c(times, rep(TIMES[i],length(Error_Rate_abs[[i]])))
}
df2 <- data.frame(Square_Error=dfes,TIMES=as.factor(times))
ggplot(df2, aes(x=Square_Error, colour = TIMES)) + geom_density() + xlim(0,1e-6) + ggtitle('Distribution of Square Error for Different Times')
