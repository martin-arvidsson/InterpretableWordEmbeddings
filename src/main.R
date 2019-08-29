# -----------------------------------------------------------------
#
#
#       Interpretable Word Embeddings via Informative Priors
#                         >> Main >>
#
#
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# (1) Set base-level of indices in tensorflow to 0 (Python style)
# -----------------------------------------------------------------
options(tensorflow.extract.one_based = FALSE)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# (2) Path where repo is stored
# -----------------------------------------------------------------
src_folder <- 'C:/Users/arvid/OneDrive/Dokument/InterpretableWordEmbeddings/src/'
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# (3) Load functions and parameters
# -----------------------------------------------------------------
source(paste0(src_folder,'data.R'))
source(paste0(src_folder,'init.R'))
source(paste0(src_folder,'utils.R'))
source(paste0(src_folder,'model.R'))
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# (4) Download and clean random sample of Gutenberg books
# -----------------------------------------------------------------
# DL
dl_sample_gutenberg(n = 5, category = 'Bestsellers, American, 1895-1923')
# Rudimentary cleaning
works_sample <- basic_cleaning(mystring = works_sample)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# (5) Process data for embedding model
# - Replace words with integer ids
# - Derive vocabulary statistics
# - Calculate negative-sampling-probabilities for each term
# -----------------------------------------------------------------
d <- process_text_string(text_string = works_sample, 
                         vocab_size = args$L)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# (6) Specify Prior Dictionary
# -----------------------------------------------------------------

# Set prior-type
# > 1 = Strict Standard Basis
# > 2 = Weak Standard Basis, 
# > 3 = 2 + Neutral, 
# > 4 = Truncated + Neutral
args$prior_type <- 3

# Example: Gender
pos = c('he', 'son', 'his', 'him', 'father',  'boy', 'himself', 'male', 'brother', 'men', 'uncl', 'nephew')
neg = c('she', 'daughter','femen','her', 'mother', 'girl', 'herself', 'femal', 'sister', 'women','aunt', 'niec')    
nltk_stopwords <- c('the','it', 'a','an','and','as','of','at','by')

# Construct list storing information about prior
args$prior_list <- list('prior_type' = args$prior_type,            # 1 = Strict Standard Basis, 2 = Weak Standard Basis, 3 = 2 + Neutral , 4 = Truncated + Neutral
                        'categ1' = pos,                            # Positive priors word types
                        'categ2' = neg,                            # Negative prior word types
                        'categ3' = nltk_stopwords,                 # Neutral word types
                        'vectors' = args$vectors,                  # On which vectors to place informative priors: rho, alpha             
                        'testset_pos' = NULL,
                        'testset_neg' = NULL)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# (7) Reorganize word-ids so that anchor word types attain ordered
#     ids, starting from 0 [this simplifies instantiation of priors]
# -----------------------------------------------------------------
objs <- reset_d_based_on_anchored_priors(args = args, 
                                         d = d, 
                                         prior_type = args$prior_type)
d <- objs$d ; args <- objs$args ; rm(objs)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# (8) Fit embedding model
# -----------------------------------------------------------------

# Run
output <- anchored_embedding(args = args, d = d)

# Extract embedding & context vectors
rho <- cbind(d$dictionary, as.data.table(output$output$rho))
alpha <- cbind(d$dictionary, as.data.table(output$output$alpha))

# Reset graph
gc() ; tf$reset_default_graph()

# Convergence?
plot(output$output$loss)
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# (9) Inspect output
# -----------------------------------------------------------------

# Identify most gendered word types
rho_gender_dim <- rho[,.(w,V100)][order(V100,decreasing = T)]
male_words <- rho_gender_dim[,head(.SD,30)][,.(w,V100,type='male')]
female_words <- rho_gender_dim[,tail(.SD,30)][,.(w,V100,type='female')]
most_gendered_words <- rbindlist(list(male_words,female_words),use.names = T, fill = T)

# Exclude anchored word types
most_gendered_words <- most_gendered_words[! w %in% unlist(args$prior_list)]

# Plot 
png('gutenberg_gender_dim.png',width=600,height=300)
ggplot(most_gendered_words,aes(x=V100,y='0',label=w)) + 
  geom_point(aes(col=factor(type))) +
  geom_vline(xintercept = 0) + 
  geom_text_repel(nudge_y=0.40,
                  segment.size  = 0.05,
                  segment.color = "grey50",
                  direction     = "x",
                  angle=50,
                  size=5) +
  scale_color_manual(breaks = c('-','','+'),
                     values = c("orange", "grey", "purple")) + 
  xlab(expression(rho^K)) +
  ylab('') +
  theme_bw(base_size = 20) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())
dev.off()
# -----------------------------------------------------------------
