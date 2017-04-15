library(foreign)
library(car)
library(corrplot)
library(visreg)
library(DAAG)
library(fmsb)
library(plyr)
library(MASS)
data = read.arff('C:/Users/karth/Documents/dataset_2216_machine_cpu.arff')
head(data, 5)
summary(data)
plot(data[, c(1:7)], pch=16, col='brown', main='Matrix Scatterplot of MYCT, MMIN, MMAX, CACH, CHMIN, CHMAX AND class')
data = data.frame(scale(data))
#mod0 is default
mod0 <- lm(class~., data=data)
summary(mod0)
#mod1: Remove CHMIN
mod1 <- lm(class ~ MYCT + MMIN + MMAX + CACH + CHMAX, data=data)
summary(mod1)
MSE <- mean((resid(mod1))^2)
coef(mod1)
press(mod1)
confint(mod1)
VIF(mod1)
#mod2 is without scaling
mod2 <- lm(class ~., data=read.arff('C:/Users/karth/Documents/dataset_2216_machine_cpu.arff'))
VIF(mod2)
data2 <- read.arff('C:/Users/karth/Documents/dataset_2216_machine_cpu.arff')
data2 = data.frame(scale(data2, center=TRUE, scale=FALSE))
#mod3 is with centering, without scaling
mod3 <- lm(class~., data=data2)
VIF((mod3))
#Residual Analysis
qqnorm(resid(mod0), pch=16)
qqline(resid(mod0))
plot(fitted(mod0), resid(mod1), pch=16)
plot(fitted(mod0), rstudent(mod1), pch=16)
Pred_R_Sq <- 1-((press(mod2)/sum(anova(mod2)$'Sum Sq')))
bc <- boxCox(mod2, lambda=seq(-2, 2, 0.1))
lambda = bc$x[which.max(bc$y)]
#Transformation
data_trans = data2
data_trans$class = data_trans$class^lambda


#Fitting and analyzing w/o scaling and centering
datanew = read.arff('C:/Users/karth/Documents/dataset_2216_machine_cpu.arff')
modnew = lm(class~ MYCT + MMIN + MMAX + CACH + CHMAX, data=datanew)
summary(modnew)
MSE <- mean((resid(modnew))^2)
coef(modnew)
press(modnew)
confint(modnew)
VIF(modnew)
#Residual Analysis of New Model
qqnorm(resid(modnew), pch=16)
qqline(resid(modnew))
plot(fitted(modnew), resid(modnew), pch=16)
plot(fitted(modnew), rstudent(modnew), pch=16)
Pred_R_Sq <- 1-((press(modnew)/sum(anova(modnew)$'Sum Sq')))
bc <- boxCox(modnew, lambda=seq(-2, 2, 0.1))
lambda = bc$x[which.max(bc$y)]
data_trans = datanew
data_trans$class = data_trans$class^lambda
mod_trans <- lm(class ~ MYCT + MMIN + MMAX + CACH + CHMAX, data=data_trans)
summary(mod_trans)
MSE_trans <- mean((resid(mod_trans))^2)
coef(mod_trans)
press(mod_trans)
confint(mod_trans)
VIF(mod_trans)
#Residual Analysis
qqnorm(resid(mod_trans), pch=16)
qqline(resid(mod_trans))
plot(fitted(mod_trans), resid(mod_trans), pch=16)
plot(fitted(mod_trans), rstudent(mod_trans), pch=16)
Pred_R_Sq_trans <- 1-((press(mod_trans)/sum(anova(mod_trans)$'Sum Sq')))
