library(gridExtra)
library(ggplot2)
library(tidyverse)
library(ggpubr)
library(lme4)
library(viridis)


####################################
########### Fongbe ############
####################################

fongbe <- read.csv('fongbe_regression.txt', header = T, sep = '\t')
fongbe$Language <- rep('Yorem Nokki', nrow(fongbe))

fongbe <- subset(fongbe,  Index %in% c('Same best model', 'Same model ranking', 'A best model', 'A best model ranking'))
fongbe$Index[fongbe$Index == 'Same model ranking'] <- 'Same ranking'
fongbe$Index[fongbe$Index == 'A best model ranking'] <- 'A best ranking'
fongbe$Index <- factor(fongbe$Index, levels = c('Same best model', 
                                            'Same ranking', 
                                            'A best model', 
                                            'A best ranking'))

fongbe_p <-
  ggplot(subset(fongbe, Metric == 'F1'&Index %in% c('Same best model', 'Same ranking')), aes(Index, Proportion, fill = Index)) +
  geom_bar(stat = 'identity', alpha = 0.8) +
  geom_text(aes(label=paste(Proportion, '%')), vjust=-1, color="black", size=6)+
  scale_fill_manual(values = c("steelblue", "mediumpurple4")) + #, "darkgreen", "peru")) +
  facet_grid(Replacement ~ Size) +
  
  theme_classic() + 
  theme(text = element_text(size=30, family="Times"),
        axis.text.x=element_blank(),
        axis.text.y=element_text(size=30)) + 
  theme(legend.position="top") +
  ylim(c(0, 100)) +
  xlab("") + 
  ylab("Proportion (%)") +
  guides(fill = guide_legend(nrow = 1)) +
  labs(fill = "") +
  ggtitle('fongbe')


########## Select one language to plot different models ###############

fongbe_details <- read.csv('fongbe_details.txt', header = T, sep = '\t')
fongbe_details <- subset(fongbe_details, Model != 'Morfessor')
fongbe_details <- subset(fongbe_details, Size == 500 & Metric == 'F1') # & Replacement == 'with')


fongbe_details %>%
  ggplot(aes(Split, Value, group = Model, color = Model)) +
  geom_line(aes(linetype=Model), alpha = 1) +
  scale_color_manual(values = c("steelblue", "peru", "darkgreen", "darkgrey", "mediumpurple4", "darkred", "black")) +
  scale_x_continuous(breaks=seq(1, 51, 5)) +
  facet_grid( ~ Replacement) +
  theme_classic() + 
  theme(text = element_text(size=16, family="Times")) + 
  theme(legend.position="top") +
  xlab("Data set") + 
  ylab("F1") + 
  xlim(c(1,50)) +
  guides(linetype = guide_legend(nrow = 2)) +
  ggtitle('fongbe 500 F1')


### The variance in the F1 for each model type across data sets ###

ggplot(subset(fongbe_details, Metric == 'F1'), aes(x = Value, color = Replacement)) +
  geom_density() + 
  facet_grid(Model ~ Size) +
  scale_color_manual(values=c("#69b3a2", "#404080")) +
  theme_classic() + 
  theme(text = element_text(size=15, family="Times"),
        axis.text.x=element_text(size=15),
        axis.text.y=element_text(size=15)) + 
  theme(legend.position="top") +
  labs(fill = "") +
  ggtitle('Yorem Nokki variance in F1')


### Calculating a breakdown ####

samples = unique(fongbe_details$Size)

fongbe_breakdown <- data.frame(Language=character(), Sample=character(), Replacement=character(), Metric=character(), Model=character(), Proportion=numeric())


for (sample in as.vector(samples)){
  for (replacement in c('with', 'without')){
    
    zero_CRF = 0
    first_CRF = 0
    second_CRF = 0
    third_CRF = 0
    fourth_CRF = 0
    seq = 0
    
    for (i in 1:50){
      
      data <- subset(fongbe_details,Split==as.character(i) & Size == sample & Replacement==replacement & Metric=='F1')
      best <- subset(data, Value == max(data$Value))
      print(best)
      if (best$Model == '0-CRF'){
        zero_CRF = zero_CRF + 1
      }
      
      if (best$Model == '1-CRF'){
        first_CRF = first_CRF + 1
      }
      
      if (best$Model == '2-CRF'){
        second_CRF = second_CRF + 1
      }
      
      if (best$Model == '3-CRF'){
        third_CRF = third_CRF + 1
      }
      
      if (best$Model == '4-CRF'){
        fourth_CRF = fourth_CRF + 1
      }
      
      if (best$Model == 'Seq2seq'){
        seq = seq + 1
      }}
    
    zero_CRF = zero_CRF * 100 / 50
    first_CRF = first_CRF * 100 / 50
    second_CRF = second_CRF * 100 / 50
    third_CRF = third_CRF * 100 / 50
    fourth_CRF = fourth_CRF * 100 / 50
    seq = seq * 100 / 50
    
    fongbe_breakdown[nrow(fongbe_breakdown) + 1, ] <- c(language, sample, replacement, 'F1', '0-CRF', zero_CRF)
    fongbe_breakdown[nrow(fongbe_breakdown) + 1, ] <- c(language, sample, replacement, 'F1', '1-CRF', first_CRF)
    fongbe_breakdown[nrow(fongbe_breakdown) + 1, ] <- c(language, sample, replacement, 'F1', '2-CRF', second_CRF)
    fongbe_breakdown[nrow(fongbe_breakdown) + 1, ] <- c(language, sample, replacement, 'F1', '3-CRF', third_CRF)
    fongbe_breakdown[nrow(fongbe_breakdown) + 1, ] <- c(language, sample, replacement, 'F1', '4-CRF', fourth_CRF)
    fongbe_breakdown[nrow(fongbe_breakdown) + 1, ] <- c(language, sample, replacement, 'F1', 'Seq2seq', seq)
    
  }}

write.csv(fongbe_breakdown, 'fongbe_breakdown.txt', row.names = FALSE)


fongbe_breakdown %>%
  ggplot(aes(Model, as.numeric(Proportion), fill = Model)) +
  geom_bar(stat = 'identity', alpha = 0.8) +
  geom_text(aes(label=Proportion), vjust=-2.6, color="black", size=3.5)+
  facet_grid(Replacement ~ Sample) +
  theme_classic() + 
  theme(text = element_text(size=30, family="Times"),
        axis.text.x=element_blank(),
        axis.text.y=element_text(size=30)) + 
  theme(legend.position="none") +
  ylim(c(0, 100)) +
  xlab("") + 
  ylab("Proportion (%)") +
  guides(fill = guide_legend(nrow = 2)) +
  labs(fill = "") +
  ggtitle('Yorem Nokki model statistics')


####### Studying the effects of data characteristics / heuristics #######
####### For original train / test random splits #########

fongbe_heuristics <- read.csv('fongbe_heuristics.txt', header = T, sep = '\t')
fongbe_full <- read.csv('fongbe_full.txt', header = T, sep = '\t')

samples = unique(fongbe_heuristics$Sample)

language <- unique(fongbe_heuristics$Language)

fongbe_df <- data.frame(Language=character(), Sample=character(), Replacement=character(), Metric=character(), Model=character(), Feature=character(), Coef=numeric(), Q2.5=numeric(), Q97.5=numeric()) 

for (sample in as.vector(samples)){
  for (replacement in c('with', 'without')){
    for (metric in c('Accuracy', 'Precision', 'Recall', 'F1', 'Avg. Distance')){
      for (model in c('Morfessor', '0-CRF', '1-CRF', '2-CRF', '3-CRF', '4-CRF', 'Seq2seq')){
        for (feature in c('word_overlap', 'morph_overlap', 
                          'ave_num_morph_ratio', 'dist_ave_num_morph', 'ave_morph_len_ratio')){
          
          heuristics <- subset(fongbe_heuristics, Feature == feature & Sample == sample & Replacement == replacement)
          results <- subset(fongbe_full, Metric == metric & Model == model & Size == sample & Replacement == replacement)
          together = cbind(results, heuristics)
          #          spearman_c = cor(together$Score, together$Value, method = c('spearman'))
          
          #          regression <- brm(Score ~ Value,
          #                              data=together,
          #                              warmup=200,
          #                              iter = 1000,
          #                              chains = 4,
          #                              inits="random",
          #                              prior=prior,
          #                              control = list(adapt_delta = 0.99),
          #                              cores = 2)
          
          #            summary <- data.frame(fixef(regression))
          
          regression <- lm(Score ~ Value, data = together)
          summary <- data.frame(summary(regression)$coef)
          coef = round(summary$Estimate[2], 2)
          ci = data.frame(confint(regression, 'Value', level = 0.95))
          q2.5 = round(ci$X2.5..[1], 2)
          q97.5 = round(ci$X97.5..[1], 2)
          
          #            q2.5 = summary$Q2.5[2]
          #            q97.5 = summary$Q97.5[2]
          
          fongbe_df[nrow(fongbe_df) + 1, ] <- c(language, sample, replacement, metric, model, feature, coef, q2.5, q97.5)
          
          fongbe_df[is.na(fongbe_df)] <- 0
          
          write.csv(fongbe_df, 'fongbe_corr.txt',row.names=FALSE)
          
        }}}}}


ggplot(subset(fongbe_df, Sample=='1000'&Replacement=='with'& (Q2.5 > 0 | Q97.5 < 0)), aes(Feature, as.numeric(Coef), fill = Feature)) +
  geom_bar(stat = 'identity', alpha = 0.8) +
  geom_errorbar(aes(ymax = as.numeric(Q97.5), ymin = as.numeric(Q2.5)), width=.1, position=position_dodge(.9)) +
  geom_text(aes(label=Coef), vjust=2.6, color="black", size=3.5) +
  facet_grid(Model ~ Metric) +
  theme_classic() + 
  theme(text = element_text(size=10, family="Times"),
        axis.text.x=element_blank(),
        axis.text.y=element_text(size=10)) + 
  theme(legend.position="top") +
  ylim(c(-1, 1)) +
  ylab("Spearman") +
  guides(fill = guide_legend(nrow = 2)) +
  labs(fill = "") +
  ggtitle('Yorem Nokki characteristics 1000 with')


together = 0

for (sample in as.vector(samples)){
  for (replacement in as.vector(unique(fongbe_heuristics$Replacement))){
    for (metric in c('Accuracy', 'Precision', 'Recall', 'F1', 'Avg. Distance')){
      for (model in c('Morfessor', '0-CRF', '1-CRF', '2-CRF', '3-CRF', '4-CRF', 'Seq2seq')){
        
        results <- subset(fongbe_full, Metric == metric & Model == model & Size == sample & Replacement == replacement)
        heuristics <- subset(fongbe_heuristics, Feature == 'word_overlap' & Sample == sample & Replacement == replacement)
        heuristics <- subset(heuristics, select = -Feature)
        names(heuristics) <- c('Language', 'Sample', 'Replacement', 'Split', 'Set', 'word_overlap', 'Caveat')
        
        for (feature in c('morph_overlap',
                          'ave_num_morph_ratio', 'dist_ave_num_morph', 'ave_morph_len_ratio')){
          
          
          if (feature == 'morph_overlap'){
            heuristics$morph_overlap <- subset(fongbe_heuristics, Feature == feature & Sample == sample & Replacement == replacement)$Value
          }
          
          if (feature == 'ave_num_morph_ratio'){
            heuristics$ave_num_morph_ratio <- subset(fongbe_heuristics, Feature == feature & Sample == sample & Replacement == replacement)$Value
          }
          
          if (feature == 'dist_ave_num_morph'){
            heuristics$dist_ave_num_morph <- subset(fongbe_heuristics, Feature == feature & Sample == sample & Replacement == replacement)$Value
          }
          
          if (feature == 'ave_morph_len_ratio'){
            heuristics$ave_morph_len_ratio <- subset(fongbe_heuristics, Feature == feature & Sample == sample & Replacement == replacement)$Value
          }
          
        }
        
        together <- rbind(together, cbind(results, heuristics))
        
        
      }}}}

together <- subset(together, Language != 0)
together$Sample <- as.numeric(together$Sample)


regression <- lm(Score ~ word_overlap * morph_overlap * ave_num_morph_ratio * dist_ave_num_morph * ave_morph_len_ratio * Replacement * Sample * Model * Metric, data = together)

regression <- lm(Score ~ word_overlap + morph_overlap + ave_num_morph_ratio + dist_ave_num_morph + ave_morph_len_ratio + Replacement + Sample + Model + Metric, data = together)

regression <- lm(Score ~ (word_overlap + morph_overlap + ave_num_morph_ratio + dist_ave_num_morph + ave_morph_len_ratio)*Replacement + (word_overlap + morph_overlap + ave_num_morph_ratio + dist_ave_num_morph + ave_morph_len_ratio)*Sample + Model + Metric, data = together)

summary <- data.frame(summary(regression)$coef)
summary$Factor <- rownames(summary)

fongbe_df <- data.frame(Language=character(), Factor=character(), Coef=numeric(), Q2.5=numeric(), Q97.5=numeric()) 


for (factor in as.vector(summary$Factor)){
  print(factor)
  coef = subset(summary, Factor == factor)$Estimate
  ci = data.frame(confint(regression, factor, level = 0.95))
  q2.5 = ci$X2.5..[1]
  q97.5 = ci$X97.5..[1]
  print(typeof(factor))
  fongbe_df[nrow(fongbe_df) + 1, ] <- c('fongbe', as.character(factor), round(coef, 2), round(q2.5, 2), round(q97.5, 2))
  write.csv(fongbe_df, 'fongbe_corr_overall.txt', row.names = FALSE)
  
}

fongbe_df$Factor<-summary$Factor
fongbe_df$P_value<-summary$Pr...t..
fongbe_df$Language<-rep('fongbe',nrow(fongbe_df))


for (feature in c('word_overlap', 'morph_overlap', 
                  'ave_num_morph_ratio', 'dist_ave_num_morph', 'ave_morph_len_ratio')){
  
  data <- subset(together, Model != 'Morfessor' & Feature == feature & Score != 0 & Value != 0)
  print(feature)
  regression <- 0
  
  if (feature %in% c('word_overlap')){
    regression <- lm(Score ~ Value * Sample * Model * Metric, data = data)
    
  }
  else {
    regression <- lm(Score ~ Value * Replacement * Sample * Model * Metric, data = data) 
    
  }
  
  summary <- data.frame(summary(regression)$coef)
  summary$Factor <- rownames(summary)
  
  for (factor in as.vector(summary$Factor)){
    coef = subset(summary, Factor == factor)$Estimate
    ci = data.frame(confint(regression, factor, level = 0.95))
    q2.5 = ci$X2.5..[1]
    q97.5 = ci$X97.5..[1]
    fongbe_df[nrow(fongbe_df) + 1, ] <- c(language, feature, factor, round(coef, 2), round(q2.5, 2), round(q97.5, 2))
    write.csv(fongbe_df, 'fongbe_corr_overall.txt', row.names = FALSE)
    
  }
  
}
