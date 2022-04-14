library(gridExtra)
library(ggplot2)
library(tidyverse)
library(ggpubr)
library(lme4)
library(viridis)


####################################
########### Fongbe ############
####################################

fongbe <- read.csv('data/fongbe/fongbe_regression.txt', header = T, sep = '\t')

duration <- subset(fongbe, Evaluation == 'len_different')
sum(duration$Duration) / 3600
min(duration$Duration)
max(duration$Train_Duration)

pitch <- subset(fongbe, Evaluation == 'ave_pitch')
sum(pitch$Duration) / 3600
min(pitch$Pitch)
max(pitch$Train_Pitch)

intensity <- subset(fongbe, Evaluation == 'ave_intensity')
sum(intensity$Duration) / 3600
min(intensity$Intensity)
max(intensity$Train_Intensity)

num_word <- subset(fongbe, Evaluation == 'num_word')
sum(num_word$Duration) / 3600
min(num_word$Num_word)
max(num_word$Train_Num_word)

word_type <- subset(fongbe, Evaluation == 'word_type')
sum(word_type$Duration) / 3600
min(word_type$Word_type)
max(word_type$Train_Word_type)

ppl<- subset(fongbe, Evaluation == 'ppl')
sum(ppl$Duration) / 3600
min(ppl$PPL)
max(ppl$Train_PPL)

distance <- subset(fongbe, Evaluation == 'distance')
unique(distance$Output)
distance_system1 <- subset(distance, Output == 'system1')
distance_system2 <- subset(distance, Output == 'system2')
distance_system3 <- subset(distance, Output == 'system3')
distance_system4 <- subset(distance, Output == 'system4')
distance_system5 <- subset(distance, Output == 'system5')
sum(distance_system1$Duration) / 3600
sum(distance_system2$Duration) / 3600
sum(distance_system3$Duration) / 3600
sum(distance_system4$Duration) / 3600
sum(distance_system5$Duration) / 3600

####################################
########### Wolof ############
####################################

wolof <- read.csv('data/wolof/wolof_regression.txt', header = T, sep = '\t')

duration <- subset(wolof, Evaluation == 'len_different')
sum(duration$Duration) / 3600
min(duration$Duration)
max(duration$Train_Duration)

pitch <- subset(wolof, Evaluation == 'ave_pitch')
sum(pitch$Duration) / 3600
min(pitch$Pitch)
max(pitch$Train_Pitch)

intensity <- subset(wolof, Evaluation == 'ave_intensity')
sum(intensity$Duration) / 3600
min(intensity$Intensity)
max(intensity$Train_Intensity)

num_word <- subset(wolof, Evaluation == 'num_word')
sum(num_word$Duration) / 3600
min(num_word$Num_word)
max(num_word$Train_Num_word)

word_type <- subset(wolof, Evaluation == 'word_type')
sum(word_type$Duration) / 3600
min(word_type$Word_type)
max(word_type$Train_Word_type)

ppl<- subset(wolof, Evaluation == 'ppl')
sum(ppl$Duration) / 3600
min(ppl$PPL)
max(ppl$Train_PPL)

distance <- subset(wolof, Evaluation == 'distance')
unique(distance$Output)
distance_system1 <- subset(distance, Output == 'system1')
distance_system2 <- subset(distance, Output == 'system2')
distance_system3 <- subset(distance, Output == 'system3')
distance_system4 <- subset(distance, Output == 'system4')
distance_system5 <- subset(distance, Output == 'system5')
sum(distance_system1$Duration) / 3600
sum(distance_system2$Duration) / 3600
sum(distance_system3$Duration) / 3600
sum(distance_system4$Duration) / 3600
sum(distance_system5$Duration) / 3600


####################################
########### Swahili ############
####################################

swahili <- read.csv('data/swahili/swahili_regression.txt', header = T, sep = '\t')

duration <- subset(swahili, Evaluation == 'len_different')
sum(duration$Duration) / 3600
min(duration$Duration)
max(duration$Train_Duration)

pitch <- subset(swahili, Evaluation == 'ave_pitch')
sum(pitch$Duration) / 3600
min(pitch$Pitch)
max(pitch$Train_Pitch)

intensity <- subset(swahili, Evaluation == 'ave_intensity')
sum(intensity$Duration) / 3600
min(intensity$Intensity)
max(intensity$Train_Intensity)

num_word <- subset(swahili, Evaluation == 'num_word')
sum(num_word$Duration) / 3600
min(num_word$Num_word)
max(num_word$Train_Num_word)

word_type <- subset(swahili, Evaluation == 'word_type')
sum(word_type$Duration) / 3600
min(word_type$Word_type)
max(word_type$Train_Word_type)

ppl<- subset(swahili, Evaluation == 'ppl')
sum(ppl$Duration) / 3600
min(ppl$PPL)
max(ppl$Train_PPL)

distance <- subset(swahili, Evaluation == 'distance')
unique(distance$Output)
distance_system1 <- subset(distance, Output == 'system1')
distance_system2 <- subset(distance, Output == 'system2')
distance_system3 <- subset(distance, Output == 'system3')
distance_system4 <- subset(distance, Output == 'system4')
distance_system5 <- subset(distance, Output == 'system5')
sum(distance_system1$Duration) / 3600
sum(distance_system2$Duration) / 3600
sum(distance_system3$Duration) / 3600
sum(distance_system4$Duration) / 3600
sum(distance_system5$Duration) / 3600

####################################
########### Iban ############
####################################

iban <- read.csv('data/iban/iban_regression.txt', header = T, sep = '\t')

duration <- subset(iban, Evaluation == 'len_different')
sum(duration$Duration) / 3600
min(duration$Duration)
max(duration$Train_Duration)

pitch <- subset(iban, Evaluation == 'ave_pitch')
sum(pitch$Duration) / 3600
min(pitch$Pitch)
max(pitch$Train_Pitch)

intensity <- subset(iban, Evaluation == 'ave_intensity')
sum(intensity$Duration) / 3600
min(intensity$Intensity)
max(intensity$Train_Intensity)

num_word <- subset(iban, Evaluation == 'num_word')
sum(num_word$Duration) / 3600
min(num_word$Num_word)
max(num_word$Train_Num_word)

word_type <- subset(iban, Evaluation == 'word_type')
sum(word_type$Duration) / 3600
min(word_type$Word_type)
max(word_type$Train_Word_type)

ppl<- subset(iban, Evaluation == 'ppl')
sum(ppl$Duration) / 3600
min(ppl$PPL)
max(ppl$Train_PPL)

distance <- subset(iban, Evaluation == 'distance')
unique(distance$Output)
distance_system1 <- subset(distance, Output == 'system1')
distance_system2 <- subset(distance, Output == 'system2')
distance_system3 <- subset(distance, Output == 'system3')
distance_system4 <- subset(distance, Output == 'system4')
distance_system5 <- subset(distance, Output == 'system5')
sum(distance_system1$Duration) / 3600
sum(distance_system2$Duration) / 3600
sum(distance_system3$Duration) / 3600
sum(distance_system4$Duration) / 3600
sum(distance_system5$Duration) / 3600


####################################
########### Hupa (verified) ############
####################################

hupa <- read.csv('data/hupa/hupa_top_tier_regression.txt', header = T, sep = '\t')

duration <- subset(hupa, Evaluation == 'len')
sum(duration$Duration) / 3600
min(duration$Duration)
max(duration$Train_Duration)

pitch <- subset(hupa, Evaluation == 'ave_pitch')
sum(pitch$Duration) / 3600
min(pitch$Pitch)
max(pitch$Train_Pitch)

intensity <- subset(hupa, Evaluation == 'ave_intensity')
sum(intensity$Duration) / 3600
min(intensity$Intensity)
max(intensity$Train_Intensity)

num_word <- subset(hupa, Evaluation == 'num_word')
sum(num_word$Duration) / 3600
min(num_word$Num_word)
max(num_word$Train_Num_word)

word_type <- subset(hupa, Evaluation == 'word_type')
sum(word_type$Duration) / 3600
min(word_type$Word_type)
max(word_type$Train_Word_type)

ppl<- subset(hupa, Evaluation == 'ppl')
sum(ppl$Duration) / 3600
min(ppl$PPL)
max(ppl$Train_PPL)

distance <- subset(hupa, Evaluation == 'distance')
unique(distance$Output)
distance_system1 <- subset(distance, Output == 'system1')
distance_system2 <- subset(distance, Output == 'system2')
distance_system3 <- subset(distance, Output == 'system3')
distance_system4 <- subset(distance, Output == 'system4')
distance_system5 <- subset(distance, Output == 'system5')
sum(distance_system1$Duration) / 3600
sum(distance_system2$Duration) / 3600
sum(distance_system3$Duration) / 3600
sum(distance_system4$Duration) / 3600
sum(distance_system5$Duration) / 3600

####################################
########### Hupa (coarse) ############
####################################

hupa <- read.csv('data/hupa/hupa_second_tier_regression.txt', header = T, sep = '\t')

duration <- subset(hupa, Evaluation == 'len')
sum(duration$Duration) / 3600
min(duration$Duration)
max(duration$Train_Duration)

pitch <- subset(hupa, Evaluation == 'ave_pitch')
sum(pitch$Duration) / 3600
min(pitch$Pitch)
max(pitch$Train_Pitch)

intensity <- subset(hupa, Evaluation == 'ave_intensity')
sum(intensity$Duration) / 3600
min(intensity$Intensity)
max(intensity$Train_Intensity)

num_word <- subset(hupa, Evaluation == 'num_word')
sum(num_word$Duration) / 3600
min(num_word$Num_word)
max(num_word$Train_Num_word)

word_type <- subset(hupa, Evaluation == 'word_type')
sum(word_type$Duration) / 3600
min(word_type$Word_type)
max(word_type$Train_Word_type)

ppl<- subset(hupa, Evaluation == 'ppl')
sum(ppl$Duration) / 3600
min(ppl$PPL)
max(ppl$Train_PPL)

distance <- subset(hupa, Evaluation == 'distance')
unique(distance$Output)
distance_system1 <- subset(distance, Output == 'system1')
distance_system2 <- subset(distance, Output == 'system2')
distance_system3 <- subset(distance, Output == 'system3')
distance_system4 <- subset(distance, Output == 'system4')
distance_system5 <- subset(distance, Output == 'system5')
sum(distance_system1$Duration) / 3600
sum(distance_system2$Duration) / 3600
sum(distance_system3$Duration) / 3600
sum(distance_system4$Duration) / 3600
sum(distance_system5$Duration) / 3600


############ Other Stuff #############

fongbe_eval <- read.csv('../results/fongbe_eval.txt', header = T, sep = '\t')
summary(lm(WER~Duration,data=fongbe_eval))

iban_eval <- read.csv('../results/iban_eval.txt', header = T, sep = '\t')
summary(lm(WER~Duration,data=iban_eval))

wolof_eval <- read.csv('../results/wolof_eval.txt', header = T, sep = '\t')
summary(lm(WER~Duration,data=wolof_eval))

swahili_eval <- read.csv('../results/swahili_eval.txt', header = T, sep = '\t')
summary(lm(WER~Duration,data=swahili_eval))

hupa_top_tier_eval <- read.csv('../results/hupa_top_tier_eval.txt', header = T, sep = '\t')
summary(lm(WER~Duration,data=hupa_top_tier_eval))

hupa_second_tier_eval <- read.csv('../results/hupa_second_tier_eval.txt', header = T, sep = '\t')
summary(lm(WER~Duration,data=hupa_second_tier_eval))



