library(tm)
library(wordcloud)
library(e1071)
library(gmodels)
library(rplot)

set.seed(1234)
rm(list = ls())

## Read Data ----
filesNegDec <- list.files("./data/simp_cornell/negative/deceptive", pattern="*.txt", full.names=TRUE)
filesNegTru <- list.files("./data/simp_cornell/negative/truthful", pattern="*.txt", full.names=TRUE)
filesPosDec <- list.files("./data/simp_cornell/positive/deceptive", pattern="*.txt", full.names=TRUE)
filesPosTru <- list.files("./data/simp_cornell/positive/truthful", pattern="*.txt", full.names=TRUE)

i = 1
df_raw = data.frame(cbind("type", "sentiment", "text"), stringsAsFactors = FALSE)
names(df_raw) = c("type", "sentiment", "text")
for (file in filesNegDec) {
  df_raw[i,] = c("deceptive", "negative", readLines(file))
  i =i+1
}
for (file in filesNegTru) {
  df_raw[i,] = c("truthful", "negative", readLines(file))
  i =i+1
}
for (file in filesPosDec) {
  df_raw[i,] = c("deceptive", "positive", readLines(file))
  i =i+1
}
for (file in filesPosTru) {
  df_raw[i,] = c("truthful", "positive", readLines(file))
  i =i+1
}

df_raw = df_raw[sample(nrow(df_raw)), ]

# Examine the structure of the sms data
str(df_raw)

# Convert spam/ham to factor.
df_raw$sentiment <- factor(df_raw$sentiment)
df_raw$type <- factor(df_raw$type)

# Examine the type variable more carefully
str(df_raw$sentiment)
table(df_raw$sentiment)
str(df_raw$type)
table(df_raw$type)
table(df_raw[, c(1,2)])

df_corpus <- Corpus(VectorSource(df_raw$text))

# examine the sms corpus
print(df_corpus)
inspect(df_corpus[1:3])

# clean up the corpus using tm_map()
corpus_clean <- tm_map(df_corpus, content_transformer(tolower))
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)

# examine the clean corpus
inspect(df_corpus[1])
inspect(corpus_clean[1])

# create a document-term sparse matrix
dtm <- DocumentTermMatrix(corpus_clean)
dtm

# creating training and test datasets
train_rows = 1:round(0.7*nrow(df_raw))
test_rows = (round(0.7*nrow(df_raw))+1):nrow(df_raw)
df_raw_train <- df_raw[train_rows, ]
df_raw_test  <- df_raw[test_rows, ]

dtm_train <- dtm[train_rows, ]
dtm_test  <- dtm[test_rows, ]

corpus_train <- corpus_clean[train_rows]
corpus_test  <- corpus_clean[test_rows]

# check that the proportion of spam is similar
prop.table(table(df_raw_train$type))
prop.table(table(df_raw_test$type))

# word cloud visualization
wordFreq <- sort(colSums(as.matrix(dtm)), decreasing=TRUE)
wordcloud(names(wordFreq), wordFreq, 
          scale=c(3, 0.5),
          min.freq=20, 
          max.words=80, 
          random.order=FALSE, rot.per=.15, colors=brewer.pal(6,"Dark2"))
title(sub="All")

wordFreq <- sort(colSums(as.matrix(dtm[df_raw$type=="deceptive",])), decreasing=TRUE)
wordcloud(names(wordFreq), wordFreq, 
          scale=c(3, 0.5),
          min.freq=20, 
          max.words=80, 
          random.order=FALSE, rot.per=.15, colors=brewer.pal(6,"Dark2"))
title(sub="Deceptive")

wordFreq <- sort(colSums(as.matrix(dtm[df_raw$type=="truthful",])), decreasing=TRUE)
wordcloud(names(wordFreq), wordFreq, 
          scale=c(3, 0.5),
          min.freq=20, 
          max.words=80, 
          random.order=FALSE, rot.per=.15, colors=brewer.pal(6,"Dark2"))
title(sub="Truthful")

# indicator features for frequent words
findFreqTerms(dtm_train, 5)
dictionary <- findFreqTerms(dtm_train, 5)
dtm_model_trainset <- DocumentTermMatrix(corpus_train, list(dictionary = dictionary))
dtm_model_testset  <- DocumentTermMatrix(corpus_test, list(dictionary = dictionary))

# convert counts to a factor
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
}

# apply() convert_counts() to columns of train/test data
dtm_model_trainset <- apply(dtm_model_trainset, MARGIN = 2, convert_counts)
dtm_model_testset  <- apply(dtm_model_testset, MARGIN = 2, convert_counts)

## Step 3: Training a model on the data ----
classifier <- naiveBayes(dtm_model_trainset, df_raw_train$type)
classifier

## Step 4: Evaluating model performance ----
test_pred <- predict(classifier, dtm_model_testset)

CrossTable(test_pred, df_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

## Step 5: Improving model performance ----
classifier2 <- naiveBayes(dtm_model_trainset, df_raw_train$type, laplace = 1)
test_pred2 <- predict(classifier2, dtm_model_testset)
CrossTable(test_pred2, df_raw_test$type,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

# Step 6 : Discriminant WordCloud ----
listTable = unlist(classifier2$tables)
indSpam = 4*(1:(length(listTable)/4))
indHam = 4*(1:(length(listTable)/4))-1

indOrdSpam = sort(listTable[indSpam], decreasing = TRUE, index.return=TRUE)
indOrdHam = sort(listTable[indHam], decreasing = TRUE, index.return=TRUE)
indOrdDiv = sort(listTable[indSpam]/listTable[indHam], decreasing = TRUE, index.return=TRUE)
indOrdDivPrior = sort((classifier2$apriori["deceptive"]*listTable[indSpam])/(classifier2$apriori["truthful"]*listTable[indHam]), decreasing = TRUE, index.return=TRUE)

indFirst = 1:80
indLast = (length(indOrdDivPrior$i)):(length(indOrdDivPrior$i)-79)

wordcloud(names(classifier2$tables[indOrdDivPrior$ix[indFirst]]), indOrdDivPrior$x[indFirst], 
          scale=c(3, 0.5),
          max.words=80, 
          random.order=FALSE, rot.per=.15, colors=brewer.pal(6,"Dark2"))
title(sub="Deceptive (discriminant)")

wordcloud(names(classifier2$tables[indOrdDivPrior$ix[indLast]]), indOrdDivPrior$x[indLast], 
          scale=c(1.3, 0.2),
          max.words=80, 
          random.order=FALSE, rot.per=.15, colors=brewer.pal(6,"Dark2"))
title(sub="Truthful (discriminant)")

# Step 7 : Debug Model Probabilities ---- 
iniTest = min(test_rows)
c(min(test_rows), max(test_rows))
listTable = unlist(classifier2$tables)
classifier2$tables[[1]]
listTable[4*(1:(length(listTable)/4))-3][1] # deceptive && No
listTable[4*(1:(length(listTable)/4))-2][1] # truthful && No
listTable[4*(1:(length(listTable)/4))-1][1] # deceptive && Yes
listTable[4*(1:(length(listTable)/4))][1] # truthful && Yes
dfTable = data.frame(cbind(listTable[4*(1:(length(listTable)/4))-3], 
                           listTable[4*(1:(length(listTable)/4))-1], 
                           listTable[4*(1:(length(listTable)/4))-2], 
                           listTable[4*(1:(length(listTable)/4))]))
colnames(dfTable) = c("DecNo", "DecYes", "TruNo", "TruYes")
dfTable[1,]
classifier2$tables[[1]]

MDec = matrix(as.matrix(dfTable[,c(1,2)]), ncol=1)
MTru = matrix(as.matrix(dfTable[,c(3,4)]), ncol=1)

intersect(iniTest:nrow(df_raw) ,intersect(which(df_raw$type=="deceptive"), which(df_raw$sentiment=="negative")))[1:5]
intersect(iniTest:nrow(df_raw) ,intersect(which(df_raw$type=="truthful"), which(df_raw$sentiment=="negative")))[1:5]
intersect(iniTest:nrow(df_raw) ,intersect(which(df_raw$type=="deceptive"), which(df_raw$sentiment=="positive")))[1:5]
intersect(iniTest:nrow(df_raw) ,intersect(which(df_raw$type=="truthful"), which(df_raw$sentiment=="positive")))[1:5]
i = 1121 # iniTest=1121 d + n
i = 1122 # iniTest=1121 t + n
i = 1127 # iniTest=1121 d + p
i = 1127 # iniTest=1121 t + p
itest = 1 + i-1121
as.character(test_pred2[itest])

nwords = nrow(MDec)/2
print(df_raw$text[i])
df_raw[i,c(1,2)]
test_pred2[itest]
sum(prod(MDec[1:nwords+nwords*as.numeric(dtm_model_testset[itest,]=="Yes")]))
sum(prod(MTru[1:nwords+nwords*as.numeric(dtm_model_testset[itest,]=="Yes")]))
sum(prod(MDec[1:nwords+nwords*as.numeric(dtm_model_testset[itest,]=="Yes")]))>sum(prod(MTru[1:nwords+nwords*as.numeric(dtm_model_testset[itest,]=="Yes")]))
r.plot(1/(1+exp(-0.1*1:10000+500)))
