## Author : Anish Anand
## Written on May,2016

setwd("C:\\Users\\Anish Anand\\Desktop\\kaggle\\titanic")
train <- read.csv("train.csv")
test <- read.csv("test.csv")

# Install and load required packages 
install.packages('randomForest')
install.packages('party')
library(randomForest)
library(rpart)
library(party)
Library(Amelia)
# Join together the test and train sets 
test$Survived <- NA
combi <- rbind(train, test)

#plot the missing values using amelia
missmap(combi)

# Convert to a string
combi$Name <- as.character(combi$Name)

# Engineered variable: Title
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi$Title <- sub(' ', '', combi$Title)
# Combine small title groups
combi$Title[combi$Title %in% c("Capt","Col","Rev","Major","Jonkheer","Don")] <- 'Noble'
combi$Title[combi$Title %in%  c("Dona","the Countess","Lady")] <- 'Lady'
combi$Title[combi$Title %in%  c("Mme","Ms","Mlle")] <- 'Miss'
combi$Title[combi$Title %in%  c("Dr","Sir","Master")]<-"Mr"

# Convert to a factor
combi$Title <- factor(combi$Title)

#Priviledged fields
combi$priviledged<-ifelse(combi$Title=="Noble"|combi$Title=="Lady",1,0)
# Engineered variable: Family size
combi$FamilySize <- combi$SibSp + combi$Parch + 1

# Engineered variable: Family
combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
#unique family id to uniquely identify memebrs of same family(familysize+surname)
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] <- 'Small' #since keeping size <=2 gives around 61 factors which cannot be analysed by RANDOM FOREST

# Delete erroneous family IDs
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'

# Convert to a factor
combi$FamilyID <- factor(combi$FamilyID)
#DECK NAME
combi$Deck_name<-gsub("+\\d","",combi$Cabin)
combi$Deck_name <- sapply(combi$Deck_name, FUN=function(x) {strsplit(x, split=" ")[[1]][1]})

# Fill in Age NAs
summary(combi$Age)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize, 
                data=combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])
# Check what else might be missing
summary(combi)
# Fill in Embarked blanks
summary(combi$Embarked)
which(combi$Embarked == '')
#comparing table of cat. variables with embarked(i.e title,gender,pclass) 
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)
# Fill in Fare NAs
summary(combi$Fare)
which(is.na(combi$Fare))
combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)

# New factor for Random Forests, only allowed <32 levels, so reduce number
combi$FamilyID2 <- combi$FamilyID
# Convert back to string
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'

# And convert back to factor
combi$FamilyID2 <- factor(combi$FamilyID2)
combi$FamilyID<-NULL

# Split back into test and train sets
train <- combi[1:891,]
test <- combi[892:1309,]

# Build Random Forest Ensemble
set.seed(415)
fit <- randomForest(as.factor(Survived) ~Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2+priviledged,
                    data=train, importance=TRUE, ntree=2000)
# Look at variable importance
varImpPlot(fit)

# Now let's make a prediction and write a submission file
Prediction <- predict(fit, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstforest1.csv", row.names = FALSE)
#############################################################################################
# Build condition inference tree Random Forest
#since priviledged field did not have much importance,we will not include in further model
set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2,
               data = train, controls=cforest_unbiased(ntree=2000, mtry=3)) 

# Now let's make a prediction and write a submission file
Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "ciforest3.csv", row.names = FALSE)

############################################################################################

