library(tm)
library(wordcloud)
library(e1071)
library(gmodels)

# Total = EasyHam + HardHam + Spam 
# 9349 = 6451 + 500 + 2398
  
## Read Data ----
filesEasyHam <- list.files("./data/spamassassin_sample/easy_ham", pattern="*", full.names=TRUE)
filesHardHam <- list.files("./data/spamassassin_sample/hard_ham", pattern="*", full.names=TRUE)
filesSpam <- list.files("./data/spamassassin_sample/spam", pattern="*", full.names=TRUE)
# filesEasyHam <- list.files("./data/spamassassin/easy_ham", pattern="*", full.names=TRUE)
# filesHardHam <- list.files("./data/spamassassin/hard_ham", pattern="*", full.names=TRUE)
# filesSpam <- list.files("./data/spamassassin/spam", pattern="*", full.names=TRUE)

i=1
df_raw = data.frame(cbind("type", "text"), stringsAsFactors = FALSE)
names(df_raw) = c("type", "text")
for (file in filesEasyHam) {  
  listLines = readLines(file)
  line = 1
  while (line <= length(listLines) && listLines[line] != "") line=line+1
  line = line +1
  
  strText = ""
  while (line <= length(listLines) ) {
    strText = paste(strText, listLines[line])
    line=line+1
  }
  df_raw[i,] = c("ham", strText)
  i =i+1
}
for (file in filesHardHam) {  
  listLines = readLines(file)
  line = 1
  while (line <= length(listLines) && listLines[line] != "") line=line+1
  line = line +1
  
  strText = ""
  while (line <= length(listLines) ) {
    strText = paste(strText, listLines[line])
    line=line+1
  }
  df_raw[i,] = c("ham", strText)
  i =i+1
}
for (file in filesSpam) {  
  listLines = readLines(file)
  line = 1
  while (line <= length(listLines) && listLines[line] != "") line=line+1
  line = line +1
  
  strText = ""
  while (line <= length(listLines) ) {
    strText = paste(strText, listLines[line])
    line=line+1
  }
  df_raw[i,] = c("spam", strText)
  i =i+1
}

df_raw = df_raw[sample(nrow(df_raw)), ]
df_raw = df_raw[1:2000, ] # Filtro
rm(list=c("file", "filesEasyHam", "filesHardHam", "filesSpam", "i", "line", "listLines", "strText"))

# examine the structure of the sms data
str(df_raw)

# convert spam/ham to factor.
df_raw$type <- factor(df_raw$type)

# examine the type variable more carefully
str(df_raw$type)
table(df_raw$type)

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

wordFreq <- sort(colSums(as.matrix(dtm[df_raw$type=="spam",])), decreasing=TRUE)
wordcloud(names(wordFreq), wordFreq, 
          scale=c(3, 0.5),
          min.freq=20, 
          max.words=80, 
          random.order=FALSE, rot.per=.15, colors=brewer.pal(6,"Dark2"))
title(sub="Spam")

wordFreq <- sort(colSums(as.matrix(dtm[df_raw$type=="ham",])), decreasing=TRUE)
wordcloud(names(wordFreq), wordFreq, 
          scale=c(3, 0.5),
          min.freq=20, 
          max.words=80, 
          random.order=FALSE, rot.per=.15, colors=brewer.pal(6,"Dark2"))
title(sub="Ham")

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
indOrdDivPrior = sort((classifier2$apriori["spam"]*listTable[indSpam])/(classifier2$apriori["ham"]*listTable[indHam]), decreasing = TRUE, index.return=TRUE)

indFirst = 1:80
indLast = (length(indOrdDivPrior$i)):(length(indOrdDivPrior$i)-79)

wordcloud(names(classifier2$tables[indOrdDivPrior$ix[indFirst]]), indOrdDivPrior$x[indFirst], 
          scale=c(1.0, 0.01),
          max.words=80, 
          random.order=FALSE, rot.per=.15, colors=brewer.pal(6,"Dark2"))
title(sub="Spam (discriminant)")

wordcloud(names(classifier2$tables[indOrdDivPrior$ix[indLast]]), indOrdDivPrior$x[indLast], 
          scale=c(1.0, 0.01),
          max.words=80, 
          random.order=FALSE, rot.per=.15, colors=brewer.pal(6,"Dark2"))
title(sub="Ham (discriminant)")
