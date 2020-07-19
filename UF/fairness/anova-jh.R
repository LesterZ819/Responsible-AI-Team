# Library Initialization
library(ggplot2)
library(dplyr)
library(reshape2)

IBM <- read.csv("/Users/jimhoover/Desktop/IBM.csv")
gender_bias<-aov(IBM$prob~IBM$Gender)
anova(gender_bias)
marital_bias<-aov(IBM$prob~IBM$MaritalStatus)
anova(marital_bias)
department_bias<-aov(IBM$prob~IBM$Department)
anova(department_bias)
maritaloutput_bias<-aov(IBM$maritalstatusoutput~IBM$MaritalStatus)
anova(maritaloutput_bias)
boxplot(IBM$prob~IBM$Gender)
boxplot(IBM$prob~IBM$MaritalStatus)
boxplot(IBM$prob~IBM$Department)
prob_bias<-aov(IBM$prob~IBM$Gender*IBM$MaritalStatus)
anova(prob_bias)

# develop some visualizations to illustrate the outcome differences (i.e. potential biases)
# visualizations inspired by - http://www.sthda.com/english/wiki/ggplot2-box-plot-quick-start-guide-r-software-and-data-visualization
# diamond is the mean for the y variable for each of the groups
attach(IBM)
x = MaritalStatus
xlabel <- "Marital Status"
y = prob
ylabel <- "Probability of Attrition"
p <- ggplot(IBM, aes(x, y, color=x)) +
  geom_boxplot() + 
#  coord_flip() +
  stat_summary(fun=mean, geom = "point", shape=23, size=2, fill="black") +
  guides(fill=guide_legend(title="Marital Status")) +
  scale_color_discrete(name=xlabel) +
  labs(title = paste0(ylabel, " by ", xlabel), x=xlabel, y=ylabel)
# coord_flip()
p

# you can change this plot to look at the distributions as histograms
x = prob
xlabel <- "Probability of Attrition"
p <- ggplot(IBM, aes(prob)) +
  geom_histogram(color="black", fill="white", binwidth = 0.01) +
  facet_wrap(~MaritalStatus, dir = "v") +
  labs(title = "Attrition by Marital Status", x = "Marital Status", y = "Count")
p

# Faceted histogram with probability on the y axis
p <- ggplot(IBM, aes(prob)) +
  geom_histogram(aes(y = (..count..)/sum(..count..)),
                 color="black", fill="white", binwidth = 0.01) +
  facet_wrap(~MaritalStatus, dir = "v") +
  labs(title = "Attrition by Marital Status", x = "Marital Status", y = "Probability")
p

# single histogram with probability on the y axis
ggplot(IBM, aes(x = prob)) +  
  geom_histogram(aes(y = (..count..)/sum(..count..)))

# anova 
marital_bias <- aov(IBM$prob~IBM$MaritalStatus)
anovaObj <- anova(marital_bias)
anovaObj

# conducting the same for Gender
x = Gender
xlabel <- "Gender"
y = prob
ylabel <- "Probability of Attrition"
p <- ggplot(IBM, aes(x, y, color=x)) +
  geom_boxplot() + 
  #  coord_flip() +
  stat_summary(fun=mean, geom = "point", shape=23, size=2, fill="black") +
  guides(fill=guide_legend(title="Gender")) +
  scale_color_discrete(name=xlabel) +
  labs(title = paste0(ylabel, " by ", xlabel), x=xlabel, y=ylabel)
# coord_flip()
p

# Faceted histogram with probability on the y axis
p <- ggplot(IBM, aes(prob)) +
  geom_histogram(aes(y = (..count..)/sum(..count..)),
                 color="black", fill="white", binwidth = 0.01) +
  facet_wrap(~Gender, dir = "v") +
  labs(title = "Attrition by Gender", x = "Marital Status", y = "Probability")
p

# anova 
gender_bias <- aov(IBM$prob~IBM$Gender)
anovaObj <- anova(gender_bias)
anovaObj

detach(IBM)
