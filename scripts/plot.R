library(ggplot2)

### Fongbe ###

fongbe <- read.csv('fongbe_eval.txt', header = T, sep = '\t')
fongbe$Duration <- fongbe$Duration / 60
fongbe$Language <- rep('Fongbe', nrow(fongbe))
fongbe$Evaluation <- replace(fongbe$Evaluation, fongbe$Evaluation == 'random_different', 'random splits')
fongbe$Evaluation <- replace(fongbe$Evaluation, fongbe$Evaluation == 'distance', 'distribution distance')
fongbe$Evaluation <- replace(fongbe$Evaluation, fongbe$Evaluation == 'heldout_speaker', 'heldout speaker')
fongbe$Evaluation <- replace(fongbe$Evaluation, fongbe$Evaluation == 'ave_intensity', 'Avg. intensity')
fongbe$Evaluation <- replace(fongbe$Evaluation, fongbe$Evaluation == 'ave_pitch', 'Avg. pitch')
fongbe$Evaluation <- replace(fongbe$Evaluation, fongbe$Evaluation == 'len_different', 'audio duration')
fongbe$Evaluation <- replace(fongbe$Evaluation, fongbe$Evaluation == 'num_word', 'N of tokens')
fongbe$Evaluation <- replace(fongbe$Evaluation, fongbe$Evaluation == 'word_type', 'N of token types')
fongbe$Evaluation <- replace(fongbe$Evaluation, fongbe$Evaluation == 'ppl', 'utterance perplexity')

fongbe_variable <- subset(fongbe,Evaluation %in% c('heldout speaker', 'random splits', 'distribution distance'))
fongbe_variable$Evaluation <- factor(fongbe_variable$Evaluation, levels = c('heldout speaker', 'random splits', 'distribution distance'))
#names(fongbe_variable) <- c('Language', 'Speaker', 'Duration', 'WER', 'Data split method')

fongbe_rest <- subset(fongbe, !(Evaluation %in% c('heldout speaker', 'random splits', 'distribution distance')))
fongbe_rest$Evaluation <- factor(fongbe_rest$Evaluation, levels = c('audio duration', 'Avg. pitch', 'Avg. intensity', 'N of tokens', 'N of token types', 'utterance perplexity'))
#names(fongbe_rest) <- c('Language', 'Speaker', 'Duration', 'WER', 'Data split method')

ggplot(fongbe_variable, aes(x=Evaluation, y=WER, fill=Evaluation)) +
  geom_boxplot() +
  scale_fill_brewer(palette="Dark2") + 
  stat_summary(fun=mean, colour="darkred", geom="point", 
               shape=18, size=6, show.legend=FALSE) + 
  stat_summary(fun = mean, geom = "text", col = "black",     # Add text to plot
               vjust = -3.3, size = 10, aes(label = paste("Mean:", round(..y.., digits = 1)))) +
  ylim(c(10, 70)) + 
  xlab('') +
  labs(fill = "Data split method") + 
  theme_classic() +
  guides(linetype = guide_legend(nrow = 2)) +
  theme(legend.position="") +
  theme(legend.title = element_blank(),
        text = element_text(size=25, family="Times"),
        axis.text.x=element_text(size=25),
        axis.text.y=element_text(size=25)) +
  scale_x_discrete(guide = guide_axis(angle = 10)) 

ggplot(fongbe_rest, aes(x=Evaluation, y=WER, color=Evaluation)) +
  geom_point(size=4,  aes(shape = Evaluation)) +
  scale_color_brewer(palette="Set2") + 
  geom_text(aes(label=WER), vjust=-1, color="black", size=6) +
  ylim(c(10, 70)) + 
  xlab('') +
  labs(fill = "Data split method") + 
  theme_classic() +
  theme(legend.position="top") +
  theme(legend.title = element_blank(),
        text = element_text(size=18, family="Times"),
        axis.text.x=element_text(size=18),
        axis.text.y=element_text(size=18)) +
  scale_x_discrete(guide = guide_axis(angle = 30)) 

### Wolof ###

wolof <- read.csv('wolof_eval.txt', header = T, sep = '\t')
wolof$Duration <- wolof$Duration / 60
wolof$Language <- rep('Wolof', nrow(wolof))
wolof$Evaluation <- replace(wolof$Evaluation, wolof$Evaluation == 'random_different', 'random splits')
wolof$Evaluation <- replace(wolof$Evaluation, wolof$Evaluation == 'distance', 'distribution distance')
wolof$Evaluation <- replace(wolof$Evaluation, wolof$Evaluation == 'heldout_speaker', 'heldout speaker')
wolof$Evaluation <- replace(wolof$Evaluation, wolof$Evaluation == 'ave_intensity', 'Avg. intensity')
wolof$Evaluation <- replace(wolof$Evaluation, wolof$Evaluation == 'ave_pitch', 'Avg. pitch')
wolof$Evaluation <- replace(wolof$Evaluation, wolof$Evaluation == 'len_different', 'audio duration')
wolof$Evaluation <- replace(wolof$Evaluation, wolof$Evaluation == 'num_word', 'N of tokens')
wolof$Evaluation <- replace(wolof$Evaluation, wolof$Evaluation == 'word_type', 'N of token types')
wolof$Evaluation <- replace(wolof$Evaluation, wolof$Evaluation == 'ppl', 'utterance perplexity')

wolof_variable <- subset(wolof,Evaluation %in% c('heldout speaker', 'random splits', 'distribution distance'))
wolof_variable$Evaluation <- factor(wolof_variable$Evaluation, levels = c('heldout speaker', 'random splits', 'distribution distance'))
#names(wolof_variable) <- c('Language', 'Speaker', 'Duration', 'WER', 'Data split method')

wolof_rest <- subset(wolof, !(Evaluation %in% c('heldout speaker', 'random splits', 'distribution distance')))
wolof_rest$Evaluation <- factor(wolof_rest$Evaluation, levels = c('audio duration', 'Avg. pitch', 'Avg. intensity', 'N of tokens', 'N of token types', 'utterance perplexity'))
#names(wolof_rest) <- c('Language', 'Speaker', 'Duration', 'WER', 'Data split method')

ggplot(wolof_variable, aes(x=Evaluation, y=WER, fill=Evaluation)) +
  geom_boxplot() +
  scale_fill_brewer(palette="Dark2") + 
  stat_summary(fun=mean, colour="darkred", geom="point", 
               shape=18, size=6, show.legend=FALSE) + 
  stat_summary(fun = mean, geom = "text", col = "black",     # Add text to plot
               vjust = -3.3, size = 10, aes(label = paste("Mean:", round(..y.., digits = 1)))) +
  ylim(c(10, 70)) + 
  xlab('') +
  labs(fill = "Data split method") + 
  theme_classic() +
  guides(linetype = guide_legend(nrow = 2)) +
  theme(legend.position="") +
  theme(legend.title = element_blank(),
        text = element_text(size=25, family="Times"),
        axis.text.x=element_text(size=25),
        axis.text.y=element_text(size=25)) +
  scale_x_discrete(guide = guide_axis(angle = 10)) 

ggplot(wolof_rest, aes(x=Evaluation, y=WER, color=Evaluation)) +
  geom_point(size=4,  aes(shape = Evaluation)) +
  scale_color_brewer(palette="Set2") + 
  geom_text(aes(label=WER), vjust=-1, color="black", size=6) +
  ylim(c(10, 70)) + 
  xlab('') +
  labs(fill = "Data split method") + 
  theme_classic() +
  theme(legend.position="top") +
  theme(legend.title = element_blank(),
        text = element_text(size=18, family="Times"),
        axis.text.x=element_text(size=18),
        axis.text.y=element_text(size=18)) +
  scale_x_discrete(guide = guide_axis(angle = 30)) 


### Swahili

swahili <- read.csv('swahili_eval.txt', header = T, sep = '\t')
swahili$Duration <- swahili$Duration / 60
swahili$Language <- rep('Swahili', nrow(swahili))
swahili$Evaluation <- replace(swahili$Evaluation, swahili$Evaluation == 'random_different', 'random splits')
swahili$Evaluation <- replace(swahili$Evaluation, swahili$Evaluation == 'distance', 'distribution distance')
swahili$Evaluation <- replace(swahili$Evaluation, swahili$Evaluation == 'heldout_speaker', 'heldout speaker')
swahili$Evaluation <- replace(swahili$Evaluation, swahili$Evaluation == 'ave_intensity', 'Avg. intensity')
swahili$Evaluation <- replace(swahili$Evaluation, swahili$Evaluation == 'ave_pitch', 'Avg. pitch')
swahili$Evaluation <- replace(swahili$Evaluation, swahili$Evaluation == 'len_different', 'audio duration')
swahili$Evaluation <- replace(swahili$Evaluation, swahili$Evaluation == 'num_word', 'N of tokens')
swahili$Evaluation <- replace(swahili$Evaluation, swahili$Evaluation == 'word_type', 'N of token types')
swahili$Evaluation <- replace(swahili$Evaluation, swahili$Evaluation == 'ppl', 'utterance perplexity')

swahili_variable <- subset(swahili,Evaluation %in% c('heldout speaker', 'random splits', 'distribution distance'))
swahili_variable$Evaluation <- factor(swahili_variable$Evaluation, levels = c('heldout speaker', 'random splits', 'distribution distance'))
#names(swahili_variable) <- c('Language', 'Speaker', 'Duration', 'WER', 'Data split method')

swahili_rest <- subset(swahili, !(Evaluation %in% c('heldout speaker', 'random splits', 'distribution distance')))
swahili_rest$Evaluation <- factor(swahili_rest$Evaluation, levels = c('audio duration', 'Avg. pitch', 'Avg. intensity', 'N of tokens', 'N of token types', 'utterance perplexity'))
#names(swahili_rest) <- c('Language', 'Speaker', 'Duration', 'WER', 'Data split method')

ggplot(swahili_variable, aes(x=Evaluation, y=WER, fill=Evaluation)) +
  geom_boxplot() +
  scale_fill_brewer(palette="Dark2") + 
  stat_summary(fun=mean, colour="darkred", geom="point", 
               shape=18, size=6, show.legend=FALSE) + 
  stat_summary(fun = mean, geom = "text", col = "black",     # Add text to plot
               vjust = -3.3, size = 10, aes(label = paste("Mean:", round(..y.., digits = 1)))) +
  ylim(c(10, 70)) + 
  xlab('') +
  labs(fill = "Data split method") + 
  theme_classic() +
  guides(linetype = guide_legend(nrow = 2)) +
  theme(legend.position="") +
  theme(legend.title = element_blank(),
        text = element_text(size=25, family="Times"),
        axis.text.x=element_text(size=25),
        axis.text.y=element_text(size=25)) +
  scale_x_discrete(guide = guide_axis(angle = 10)) 

ggplot(swahili_rest, aes(x=Evaluation, y=WER, color=Evaluation)) +
  geom_point(size=4,  aes(shape = Evaluation)) +
  scale_color_brewer(palette="Set2") + 
  geom_text(aes(label=WER), vjust=-1, color="black", size=6) +
  ylim(c(10, 70)) + 
  xlab('') +
  labs(fill = "Data split method") + 
  theme_classic() +
  theme(legend.position="top") +
  theme(legend.title = element_blank(),
        text = element_text(size=15, family="Times"),
        axis.text.x=element_text(size=14),
        axis.text.y=element_text(size=14)) +
  scale_x_discrete(guide = guide_axis(angle = 30)) 


### Iban ###

iban <- read.csv('iban_eval.txt', header = T, sep = '\t')
iban$Duration <- iban$Duration / 60
iban$Language <- rep('Iban', nrow(iban))
iban$Evaluation <- replace(iban$Evaluation, iban$Evaluation == 'random_different', 'random splits')
iban$Evaluation <- replace(iban$Evaluation, iban$Evaluation == 'distance', 'distribution distance')
iban$Evaluation <- replace(iban$Evaluation, iban$Evaluation == 'heldout_speaker', 'heldout speaker')
iban$Evaluation <- replace(iban$Evaluation, iban$Evaluation == 'ave_intensity', 'Avg. intensity')
iban$Evaluation <- replace(iban$Evaluation, iban$Evaluation == 'ave_pitch', 'Avg. pitch')
iban$Evaluation <- replace(iban$Evaluation, iban$Evaluation == 'len_different', 'audio duration')
iban$Evaluation <- replace(iban$Evaluation, iban$Evaluation == 'num_word', 'N of tokens')
iban$Evaluation <- replace(iban$Evaluation, iban$Evaluation == 'word_type', 'N of token types')
iban$Evaluation <- replace(iban$Evaluation, iban$Evaluation == 'ppl', 'utterance perplexity')

iban_variable <- subset(iban,Evaluation %in% c('heldout speaker', 'random splits', 'distribution distance'))
iban_variable$Evaluation <- factor(iban_variable$Evaluation, levels = c('heldout speaker', 'random splits', 'distribution distance'))
#names(iban_variable) <- c('Language', 'Speaker', 'Duration', 'WER', 'Data split method')

iban_rest <- subset(iban, !(Evaluation %in% c('heldout speaker', 'random splits', 'distribution distance')))
iban_rest$Evaluation <- factor(iban_rest$Evaluation, levels = c('audio duration', 'Avg. pitch', 'Avg. intensity', 'N of tokens', 'N of token types', 'utterance perplexity'))
#names(iban_rest) <- c('Language', 'Speaker', 'Duration', 'WER', 'Data split method')

ggplot(iban_variable, aes(x=Evaluation, y=WER, fill=Evaluation)) +
  geom_boxplot() +
  scale_fill_brewer(palette="Dark2") + 
  stat_summary(fun=mean, colour="darkred", geom="point", 
               shape=18, size=6, show.legend=FALSE) + 
  stat_summary(fun = mean, geom = "text", col = "black",     # Add text to plot
               vjust = -3.3, size=10, aes(label = paste("Mean:", round(..y.., digits = 1)))) +
  ylim(c(10, 70)) + 
  xlab('') +
  labs(fill = "Data split method") + 
  theme_classic() +
  guides(linetype = guide_legend(nrow = 2)) +
  theme(legend.position="") +
  theme(legend.title = element_blank(),
        text = element_text(size=25, family="Times"),
        axis.text.x=element_text(size=25),
        axis.text.y=element_text(size=25)) +
  scale_x_discrete(guide = guide_axis(angle = 10)) 

hupa_top_tier <- read.csv('hupa_top_tier_eval.txt', header = T, sep = '\t')
hupa_top_tier$Duration <- hupa_top_tier$Duration / 60
hupa_top_tier$Language <- rep('Hupa (verified)', nrow(hupa_top_tier))

hupa_second_tier <- read.csv('hupa_second_tier_eval.txt', header = T, sep = '\t')
hupa_second_tier$Duration <- hupa_second_tier$Duration / 60
hupa_second_tier$Language <- rep('Hupa (coarse)', nrow(hupa_second_tier))

### All together ####

data <- rbind(fongbe, wolof, swahili, iban, hupa_top_tier, hupa_second_tier)
data$Language <- factor(data$Language, levels = c('Fongbe', 'Wolof', 'Swahili', 'Iban', 'Hupa (verified)', 'Hupa (coarse)'))

ggplot(data, aes(x=Duration, y=WER)) + 
  geom_point(size=2, aes(color = Language)) + 
  scale_color_manual(values = c("steelblue", "mediumpurple4", "darkgreen", "peru", "darkred", "darkgrey")) +
#  labs(title="Wolof",x="Audio duration (s)", y = "WER")+
  facet_wrap( ~ Language, nrow = 2) +
  theme_classic() + 
  guides(linetype = guide_legend(nrow = 2)) +
  theme(legend.position="none") +
  theme(legend.title = element_blank(),
        text = element_text(size=16, family="Times"),
        axis.text.x=element_text(size=18),
        axis.text.y=element_text(size=18)) 

