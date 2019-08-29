# -----------------------------------------------------------------
#
#
#       Interpretable Word Embeddings via Informative Priors
#         >> Initialize parameters & Load packages >>
#
#
# -----------------------------------------------------------------


options(tensorflow.extract.one_based = FALSE)


# -----------------------------------------------------------------
# Load needed packages/modules
# -----------------------------------------------------------------
library(data.table)
library(tensorflow)
library(ggplot2)
library(ggrepel)
#tfp <- import("tensorflow_probability")
#tfd <- tfp$distributions
np <- import("numpy")
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# Initialize parameters
# -----------------------------------------------------------------
args <- list()
args$datafile <- 'example_text.txt'                       # name of datafile
args$cs <- 8L                                             # context size
args$cs2 <- as.integer(args$cs/2)                         # context size
args$reduced_Ntokens <- NULL                              # reduced sample of tokens (1M instead of ~9M)       
args$L <- 10000L                                          # vocabulary size
args$K <- 100L                                            # number of embedding dimensions
args$ns <- 20L                                            # number of negative samples
args$n_iter <- 20                                         # number of passes over data
args$n_minibatch <- 10000L                                # size of minibatches
args$batch_size <- as.integer(args$n_minibatch + args$cs) # batch size (add cs to minibatch size)

# Prior specification
args$prior_type <- 1
args$vectors <- c('rho','alpha')                          # Place informative priors on both rho/alpha?

# Sigmas
args$sig <- 1.0                                           # sigma (for all dims, for all non anchor word types)
args$sigmas$sig_n <- 0.0000001                            # gamma (K'th dim for pos/neg anchor word types)
args$sigmas$sig_d_m <- 1.0                                # omega (dim 1-K for pos/neg anchor word types)
args$sigmas$sig_pn_n <- 0.001                             # psi   (K'th dim for neutral anchor word types)

# What to return
args$return_embvectors <- TRUE
args$return_loss <- TRUE
# -----------------------------------------------------------------