library(gridExtra)
library(ggplot2)
library(tidyverse)
library(ggpubr)
library(viridis)
library(olsrr)
library(lme4)
library(car)
library(MuMIn)
library(lmerTest)

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

####### Fongbe Modeling ######

fongbe$Duration_ratio <- as.numeric(fongbe$Duration_ratio)
fongbe$Pitch_ratio <- as.numeric(fongbe$Pitch_ratio)
fongbe$Intensity_ratio <- as.numeric(fongbe$Intensity_ratio)
fongbe$PPL_ratio <- as.numeric(fongbe$PPL_ratio)
fongbe$Num_word_ratio <- as.numeric(fongbe$Num_word_ratio)
fongbe$Word_type_ratio <- as.numeric(fongbe$Word_type_ratio)
fongbe$OOV_ratio <- as.numeric(fongbe$OOV_ratio)


fongbe_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio, data = fongbe)
car::vif(fongbe_model)

fongbe_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Word_type_ratio + OOV_ratio, data = fongbe)
car::vif(fongbe_model)

fongbe_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + OOV_ratio, data = fongbe)
car::vif(fongbe_model)


fongbe_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (1 | Transcript) + (Duration_ratio || Speaker), data = fongbe, control = lmerControl(check.nobs.vs.nRE = "ignore"))
summary(fongbe_model)


saveRDS(fongbe_model, file = 'fongbe_model.rds')

r.squaredGLMM(fongbe_model)

summary(lmerTest :: lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (1 | Transcript) + (Duratio_ratio || Speaker), data = fongbe, control = lmerControl(check.nobs.vs.nRE = "ignore")))

confint(fongbe_model)


prior = prior(student_t(3,0,8), class = b)

run_model <- function(expr, path, reuse = TRUE) {
  path <- paste0(path, ".Rds")
  if (reuse) {
    fit <- suppressWarnings(try(readRDS(path), silent = TRUE))
  }
  if (is(fit, "try-error")) {
    fit <- eval(expr)
    saveRDS(fit, file = path)
  }
  fit
}

fongbe_model <- run_model(brm(
  WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio + (1 | Speaker), 
  data = fongbe,
  prior = prior,
  warmup = 400,
  iter = 2000,
  chains = 4,
  init = "random",
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  cores = 6),
  path = '/data/liuaal/kaldi/fongbe_brms')

fongbe_model <- brm(
  WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio + (Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio | Speaker) + (Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio | Transcript),  
  data = fongbe,
  prior = prior,
  warmup = 400,
  iter = 2000,
  chains = 2,
  init = "random",
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  cores = 6, 
  opencl = opencl(c(0, 0)),
  backend = "cmdstanr")



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

### Modeling ###

wolof$Duration_ratio <- as.numeric(wolof$Duration_ratio)
wolof$Pitch_ratio <- as.numeric(wolof$Pitch_ratio)
wolof$Intensity_ratio <- as.numeric(wolof$Intensity_ratio)
wolof$PPL_ratio <- as.numeric(wolof$PPL_ratio)
wolof$Num_word_ratio <- as.numeric(wolof$Num_word_ratio)
wolof$Word_type_ratio <- as.numeric(wolof$Word_type_ratio)
wolof$OOV_ratio <- as.numeric(wolof$OOV_ratio)

wolof_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio, data = wolof)
car::vif(wolof_model)

wolof_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Word_type_ratio + OOV_ratio, data = wolof)
car::vif(wolof_model)

wolof_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + OOV_ratio, data = wolof)
car::vif(wolof_model)

wolof_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (1 | Transcript) + (Duration_ratio + Pitch_ratio || Speaker), data = wolof, control = lmerControl(check.nobs.vs.nRE = "ignore"))
summary(wolof_model)

saveRDS(wolof_model, file = 'wolof_model.rds')

summary(lmerTest :: lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (1 | Transcript) + (Duration_ratio + Pitch_ratio || Speaker), data = wolof, control = lmerControl(check.nobs.vs.nRE = "ignore")))

r.squaredGLMM(wolof_model)

confint(wolof_model)


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


### Modeling ###

swahili$Duration_ratio <- as.numeric(swahili$Duration_ratio)
swahili$Pitch_ratio <- as.numeric(swahili$Pitch_ratio)
swahili$Intensity_ratio <- as.numeric(swahili$Intensity_ratio)
swahili$PPL_ratio <- as.numeric(swahili$PPL_ratio)
swahili$Num_word_ratio <- as.numeric(swahili$Num_word_ratio)
swahili$Word_type_ratio <- as.numeric(swahili$Word_type_ratio)
swahili$OOV_ratio <- as.numeric(swahili$OOV_ratio)

swahili_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio, data = swahili)
car::vif(swahili_model)

swahili_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Word_type_ratio + OOV_ratio, data = swahili)
car::vif(swahili_model)

swahili_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + OOV_ratio, data = swahili)
car::vif(swahili_model)

swahili_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio | Transcript), data = swahili)

swahili_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Word_type_ratio + OOV_ratio || Transcript), data = swahili, control = lmerControl(check.nobs.vs.nRE = "ignore"))

swahili_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + OOV_ratio || Transcript), data = swahili, control = lmerControl(check.nobs.vs.nRE = "ignore"))

swahili_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio || Transcript), data = swahili, control = lmerControl(check.nobs.vs.nRE = "ignore"))

swahili_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio || Transcript), data = swahili, control = lmerControl(check.nobs.vs.nRE = "ignore")) 

swahili_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio || Transcript), data = swahili, control = lmerControl(check.nobs.vs.nRE = "ignore"))

swahili_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio  || Transcript), data = swahili, control = lmerControl(check.nobs.vs.nRE = "ignore"))

swahili_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (1  | Transcript), data = swahili, control = lmerControl(check.nobs.vs.nRE = "ignore"))

saveRDS(swahili_model, file = 'swahili_model.rds')

r.squaredGLMM(swahili_model)

summary(lmerTest :: lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio || Transcript), data = swahili, control = lmerControl(check.nobs.vs.nRE = "ignore")))

confint(swahili_model)

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

### Modeling ###

iban$Duration_ratio <- as.numeric(iban$Duration_ratio)
iban$Pitch_ratio <- as.numeric(iban$Pitch_ratio)
iban$Intensity_ratio <- as.numeric(iban$Intensity_ratio)
iban$PPL_ratio <- as.numeric(iban$PPL_ratio)
iban$Num_word_ratio <- as.numeric(iban$Num_word_ratio)
iban$Word_type_ratio <- as.numeric(iban$Word_type_ratio)
iban$OOV_ratio <- as.numeric(iban$OOV_ratio)

iban_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio, data = iban)
car::vif(iban_model)

iban_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Word_type_ratio + OOV_ratio, data = iban)
car::vif(iban_model)

iban_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + OOV_ratio, data = iban)
car::vif(iban_model)

iban_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (1 | Transcript) + (Duration_ratio + Pitch_ratio + Intensity_ratio || Speaker), data = iban, control = lmerControl(check.nobs.vs.nRE = "ignore"))
summary(iban_model)


saveRDS(iban_model, file = 'iban_model.rds')

summary(lmerTest :: lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (1 | Transcript) + (Duration_ratio + Pitch_ratio + Intensity_ratio || Speaker), data = iban, control = lmerControl(check.nobs.vs.nRE = "ignore")))

r.squaredGLMM(iban_model)

confint(iban_model)

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

### Hupa (verified) Modeling ###
### including results from heldout session ###

hupa <- read.csv('data/hupa/hupa_top_tier_full_regression.txt', header = T, sep = ',')

hupa$Duration_ratio <- as.numeric(hupa$Duration_ratio)
hupa$Pitch_ratio <- as.numeric(hupa$Pitch_ratio)
hupa$Intensity_ratio <- as.numeric(hupa$Intensity_ratio)
hupa$PPL_ratio <- as.numeric(hupa$PPL_ratio)
hupa$Num_word_ratio <- as.numeric(hupa$Num_word_ratio)
hupa$Word_type_ratio <- as.numeric(hupa$Word_type_ratio)
hupa$OOV_ratio <- as.numeric(hupa$OOV_ratio)

hupa_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio, data = hupa)
car::vif(hupa_model)

hupa_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Word_type_ratio + OOV_ratio, data = hupa)
car::vif(hupa_model)

hupa_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + OOV_ratio, data = hupa)
car::vif(hupa_model)

hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio | Transcript), data = hupa)

hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Word_type_ratio + OOV_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))

summary(hupa_model)

saveRDS(hupa_model, file = 'hupa_model.rds')

hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + OOV_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))

hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))

hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore")) 

hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))

hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio  || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))

hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (1  | Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))

r.squaredGLMM(hupa_model)

summary(lmerTest :: lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore")))

confint(hupa_model)

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


### Hupa (coarse) Modeling ###
### including results from heldout session ###

hupa <- read.csv('data/hupa/hupa_second_tier_full_regression.txt', header = T, sep = ',')

hupa$Duration_ratio <- as.numeric(hupa$Duration_ratio)
hupa$Pitch_ratio <- as.numeric(hupa$Pitch_ratio)
hupa$Intensity_ratio <- as.numeric(hupa$Intensity_ratio)
hupa$PPL_ratio <- as.numeric(hupa$PPL_ratio)
hupa$Num_word_ratio <- as.numeric(hupa$Num_word_ratio)
hupa$Word_type_ratio <- as.numeric(hupa$Word_type_ratio)
hupa$OOV_ratio <- as.numeric(hupa$OOV_ratio)

hupa_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio, data = hupa)
car::vif(hupa_model)

hupa_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Word_type_ratio + OOV_ratio, data = hupa)
car::vif(hupa_model)

hupa_model <- lm(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + OOV_ratio, data = hupa)
car::vif(hupa_model)

hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Num_word_ratio + Word_type_ratio + OOV_ratio | Transcript), data = hupa)

hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + Word_type_ratio + OOV_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))

summary(hupa_model)

saveRDS(hupa_model, file = 'hupa_model.rds')

hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio + OOV_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))
hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))
hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore")) 
hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))
hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio  || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))
hupa_model <- lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (1  | Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore"))

r.squaredGLMM(hupa_model)

summary(lmerTest :: lmer(WER ~ Evaluation + Duration_ratio + Pitch_ratio + Intensity_ratio + PPL_ratio +  Word_type_ratio + OOV_ratio + (Duration_ratio + Pitch_ratio + Intensity_ratio || Transcript), data = hupa, control = lmerControl(check.nobs.vs.nRE = "ignore")))

confint(hupa_model)

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


