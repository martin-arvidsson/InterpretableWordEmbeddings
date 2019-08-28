# -----------------------------------------------------------------
#
#
#       Interpretable Word Embeddings via Informative Priors
#                        >> Model >>
#
#
# -----------------------------------------------------------------

options(tensorflow.extract.one_based = FALSE)

anchored_embedding <- function(args, d){
  
  # ========================================
  # (1) Preliminaries
  # ========================================
  
  # - Initialize TF-session
  sess = tf$Session()
  # - Set parameters
  batch_size <- args$batch_size
  n_minibatch <- args$n_minibatch
  cs <- args$cs
  cs2 <- args$cs2
  L <- args$L
  K <- args$K
  sig <- args$sig
  ns <- args$ns
  N <- d$Ntokens
  n_iter <- args$n_iter
  unigram_value <- d$unigram$sampling_value
  m <- list()
  
  
  # ========================================
  # (2) Specify Model
  # ========================================
  
  # Placeholder is assigned "current_batch"
  # -----------------------------------------
  placeholders <- tf$placeholder(dtype = tf$int32, name = 'placeholders')
  m$words <- placeholders
  
  # Set up id-structure for center & contexts
  # -----------------------------------------
  m$p_mask <- tf$cast(x = tf$range(start = cs2, limit = n_minibatch + cs2), dtype = tf$int32) 
  rows <- tf$cast(x = tf$tile(input = tf$expand_dims(input = tf$range(start = 0L, limit = cs2), axis = list(0L)), multiples = list(n_minibatch,1L)), dtype = tf$int32)
  columns <- tf$cast(x = tf$tile(input = tf$expand_dims(input = tf$range(start = 0L, limit = n_minibatch),axis = list(1L)), multiples = list(1L,cs2)),dtype = tf$int32)
  m$ctx_mask <- tf$concat(values = list(rows+columns,rows+columns+cs2+1L),axis = 1L)
  
  # Initial values for rho/alpha
  # -----------------------------------------
  m$rho_init <- matrix(data = 0.1*rnorm(n = args$L*args$K,0,1),nrow = args$L, ncol = args$K)
  m$alpha_init <- matrix(data = 0.1*rnorm(n = args$L*args$K,0,1),nrow = args$L, ncol = args$K)
  
  # Prelims for anchored embeddings (note: updates rho_init/alpha_init)
  # -----------------------------------------
  if(args$prior_list[['prior_type']]!=0){
    m <- anchored_prior_prelims(m = m, d = d, args = args)
  }
  
  # Create tf-variables for embedding & context vectors
  # -----------------------------------------
  m$rho <- tf$Variable(initial_value = m$rho_init, name = "rho", dtype=tf$float32)
  m$alpha <- tf$Variable(initial_value = m$alpha_init, name = "alpha", dtype=tf$float32)
  
  # Clip values for Truncated Prior
  # -----------------------------------------
  if(args$prior_list[['prior_type']]==7){
    d_pos_idx <- d$prior_idx_dt[type=='d_pos_idx']$new_id
    d_neg_idx <- d$prior_idx_dt[type=='d_neg_idx']$new_id
    # RHO
    # =====
    # get ids for vectors we want to put (special) priors on
    d_pos_idx <- d$prior_idx_dt[type=='d_pos_idx']$new_id
    d_neg_idx <- d$prior_idx_dt[type=='d_neg_idx']$new_id
    # (vertical split) extract emebeddings for those indices
    rho_temp1 <- m$rho[d_pos_idx,]
    rho_temp2 <- m$rho[d_neg_idx,]
    rho_temp3 <- m$rho[(max(d_neg_idx)+1):(nrow(m$rho)-1),]
    # (horizontal split) extract columns
    # - make into matrix if we only use d_freq==1
    if(length(dim(rho_temp1))==1){rho_temp1 <- tf$reshape(tensor = rho_temp1, shape = list(1L,m$sdim1))}
    if(length(dim(rho_temp2))==1){rho_temp2 <- tf$reshape(tensor = rho_temp2, shape = list(1L,m$sdim1))}
    # - pos
    rho_temp11 <- rho_temp1[,c(0:(m$sdim0-1))]
    rho_temp12 <- rho_temp1[,m$sdim0]
    rho_temp12 <- tf$reshape(rho_temp12, list(dim(rho_temp12),1L))
    # - neg
    rho_temp21 <- rho_temp2[,c(0:(m$sdim0-1))]
    rho_temp22 <- rho_temp2[,m$sdim0]
    rho_temp22 <- tf$reshape(rho_temp22, list(dim(rho_temp22),1L))
    # (apply constraints)
    rho_temp12 <- tf$clip_by_value(rho_temp12,clip_value_min=0, clip_value_max=10.0)
    rho_temp22 <- tf$clip_by_value(rho_temp22,clip_value_min=-10.0, clip_value_max=0)
    # (horizontal merge) 
    rho_temp1 <- tf$concat(list(rho_temp11,rho_temp12),axis=1L)
    rho_temp2 <- tf$concat(list(rho_temp21,rho_temp22),axis=1L)
    # (vertical merge)
    m$rho <- tf$concat(list(rho_temp1,rho_temp2,rho_temp3),axis=0L)
    # =====
    # ALPHA
    # =====
    # get ids for vectors we want to put (special) priors on
    d_pos_idx <- d$prior_idx_dt[type=='d_pos_idx']$new_id
    d_neg_idx <- d$prior_idx_dt[type=='d_neg_idx']$new_id
    # (vertical split) extract emebeddings for those indices
    alpha_temp1 <- m$alpha[d_pos_idx,]
    alpha_temp2 <- m$alpha[d_neg_idx,]
    alpha_temp3 <- m$alpha[(max(d_neg_idx)+1):(nrow(m$alpha)-1),]
    # (horizontal split) extract columns
    # - make into matrix if we only use d_freq==1
    if(length(dim(alpha_temp1))==1){alpha_temp1 <- tf$reshape(tensor = alpha_temp1, shape = list(1L,m$sdim1))}
    if(length(dim(alpha_temp2))==1){alpha_temp2 <- tf$reshape(tensor = alpha_temp2, shape = list(1L,m$sdim1))}
    # - pos
    alpha_temp11 <- alpha_temp1[,c(0:(m$sdim0-1))]
    alpha_temp12 <- alpha_temp1[,m$sdim0]
    alpha_temp12 <- tf$reshape(alpha_temp12, list(dim(alpha_temp12),1L))
    # - neg
    alpha_temp21 <- alpha_temp2[,c(0:(m$sdim0-1))]
    alpha_temp22 <- alpha_temp2[,m$sdim0]
    alpha_temp22 <- tf$reshape(alpha_temp22, list(dim(alpha_temp22),1L))
    # (apply constraints)
    alpha_temp12 <- tf$clip_by_value(alpha_temp12,clip_value_min=0, clip_value_max=10.0)
    alpha_temp22 <- tf$clip_by_value(alpha_temp22,clip_value_min=-10.0, clip_value_max=0)
    # (horizontal merge) 
    alpha_temp1 <- tf$concat(list(alpha_temp11,alpha_temp12),axis=1L)
    alpha_temp2 <- tf$concat(list(alpha_temp21,alpha_temp22),axis=1L)
    # (vertical merge)
    m$alpha <- tf$concat(list(alpha_temp1,alpha_temp2,alpha_temp3),axis=0L)
    # =====
  }
  
  # Global priors
  # -----------------------------------------
  regular_idx <- as.integer(1:args$L)
  regular_idx <- regular_idx[!regular_idx %in% d$prior_idx_dt$new_id]
  global_prior <- tf$distributions$Normal(loc = 0, scale = sig, name = 'prior')
  m$log_prior <- tf$reduce_sum(global_prior$log_prob(m$rho[regular_idx,]) + global_prior$log_prob(m$alpha[regular_idx,]))
  
  # Informative priors
  # -----------------------------------------
  
  # // Create tf-distributions for priors //
  if(args$prior_list[['prior_type']]!=0){
    # - indices -
    d_pos_idx <- d$prior_idx_dt[type=='d_pos_idx']$new_id
    d_neg_idx <- d$prior_idx_dt[type=='d_neg_idx']$new_id
    neutral_idx <- d$prior_idx_dt[type=='n_idx']$new_id
    # - discriminative words -
    # truncated
    if(args$prior_list[['prior_type']]==7){
      d_pos_prior = tfp$distributions$TruncatedNormal(loc=0, low = 0, high = 10.0, scale=args$sigmas$sig_n)
      d_neg_prior = tfp$distributions$TruncatedNormal(loc=0, low = -10.0, high = 0, scale=args$sigmas$sig_n)
      d_zero_prior = tf$distributions$Normal(loc=0.0, scale=args$sigmas$sig_d_m)
    # standard basis
    }else{
      d_pos_prior = tf$distributions$Normal(loc=1.0, scale=args$sigmas$sig_n)
      d_neg_prior = tf$distributions$Normal(loc=-1.0, scale=args$sigmas$sig_n)
      d_zero_prior = tf$distributions$Normal(loc=0.0, scale=args$sigmas$sig_d_m)
    }
    # - neutral words -
    pn_other_prior = tf$distributions$Normal(loc=0.0, scale=args$sigmas$sig_pn_m)
    pn_zero_prior = tf$distributions$Normal(loc=0.0, scale=args$sigmas$sig_pn_n)
  }
  
  # // Instantiate priors on discrimiantive-words //
  if(args$prior_list[['prior_type']] > 0){
    if('rho' %in% args$prior_list[['vectors']] & sum(c(length(d_pos_idx),length(d_neg_idx)))>0){
      m$log_prior <- m$log_prior + tf$reduce_sum(d_pos_prior$log_prob(m$rho[d_pos_idx, m$sdim0]))
      m$log_prior <- m$log_prior + tf$reduce_sum(d_neg_prior$log_prob(m$rho[d_neg_idx, m$sdim0]))
      m$log_prior <- m$log_prior + tf$reduce_sum(d_zero_prior$log_prob(m$rho[c(d_pos_idx,d_neg_idx), 0:(m$sdim0-1)]))
    }
    if('alpha' %in% args$prior_list[['vectors']] & sum(c(length(d_pos_idx),length(d_neg_idx)))>0){
      m$log_prior <- m$log_prior + tf$reduce_sum(d_pos_prior$log_prob(m$alpha[d_pos_idx, m$sdim0]))
      m$log_prior <- m$log_prior + tf$reduce_sum(d_neg_prior$log_prob(m$alpha[d_neg_idx, m$sdim0]))
      m$log_prior <- m$log_prior + tf$reduce_sum(d_zero_prior$log_prob(m$alpha[c(d_pos_idx,d_neg_idx), 0:(m$sdim0-1)]))
    }
  }
  # // Instantiate priors on neutral-words //
  if(args$prior_list[['prior_type']] >= 3 & length(neutral_idx)>0){
    if('rho' %in% args$prior_list[['vectors']]){
      m$log_prior = m$log_prior + tf$reduce_sum(pn_zero_prior$log_prob(m$rho[neutral_idx, m$sdim0]))
      m$log_prior = m$log_prior + tf$reduce_sum(pn_other_prior$log_prob(m$rho[neutral_idx, 0:(m$sdim0-1)]))
    }
    if('alpha' %in% args$prior_list[['vectors']]){
      m$log_prior = m$log_prior + tf$reduce_sum(pn_zero_prior$log_prob(m$alpha[neutral_idx, m$sdim0]))
      m$log_prior = m$log_prior + tf$reduce_sum(pn_other_prior$log_prob(m$alpha[neutral_idx, 0:(m$sdim0-1)]))
    }
  }
  # -----------------------------------------
  
  # Collect and assign data for current batch
  # -----------------------------------------
  # - Taget and Context Indices
  m$p_idx = tf$gather(params = m$words, indices = m$p_mask, name = 'p_idx')
  m$p_rho = tf$squeeze(input = tf$gather(params = m$rho, indices = m$p_idx), name = 'p_rho')
  # - Negative samples
  unigram_logits = tf$tile(input = tf$expand_dims(input = tf$log(x = tf$constant(value = unigram_value)),axis = list(0L)),multiples = list(n_minibatch, 1L))
  m$n_idx = tf$multinomial(logits = unigram_logits, num_samples = ns)
  m$n_rho = tf$gather(params = m$rho, indices = m$n_idx) #p_idx
  # - Context
  m$ctx_idx = tf$squeeze(input = tf$gather(params = m$words, indices = m$ctx_mask))
  m$ctx_alphas = tf$gather(params = m$alpha, indices = m$ctx_idx)
  # - Natural parameter
  ctx_sum = tf$reduce_sum(m$ctx_alphas, list(1L))
  m$p_eta = tf$expand_dims(input = tf$reduce_sum(tf$multiply(x = m$p_rho, y = ctx_sum),-1L),axis = 1L)
  m$n_eta = tf$reduce_sum(tf$multiply(x = m$n_rho, y = tf$tile(input = tf$expand_dims(input = ctx_sum,axis = 1L),multiples = list(1L,ns,1L))),-1L)
  
  # Conditional likelihoods
  # -----------------------------------------
  m$y_pos = tf$distributions$Bernoulli(logits = m$p_eta)
  m$y_neg = tf$distributions$Bernoulli(logits = m$n_eta)
  m$ll_pos = tf$reduce_sum(m$y_pos$log_prob(1.0))
  m$ll_neg = tf$reduce_sum(m$y_neg$log_prob(0.0))
  m$log_likelihood = m$ll_pos + m$ll_neg
  scale = 1.0 * N / n_minibatch
  loss = - (scale * m$log_likelihood + m$log_prior)
  
  # Specify optimizer
  # -----------------------------------------
  optimizer = tf$train$AdamOptimizer()
  train = optimizer$minimize(loss,name = 'train')
  tf$global_variables_initializer()$run
  
  
  # ========================================
  # (3) Estimation
  # ========================================
  
  # Iter & epoch settings
  epochs <- floor(length(d$text_int) / batch_size)       # currently: epochs is determined by the batch-size
  
  # Initialize?
  init_op = tf$global_variables_initializer()
  sess$run(init_op)
  
  # Run
  loss_stats <- ll_stats <- c()
  k <- 0
  
  for(i in 1:n_iter){
    current_pos <- 1
    for(j in 1:epochs){

      # Select current upper-limit
      upper <- (current_pos+(batch_size+cs))
      if(upper>length(d$text_int)){
        break
      }
      
      # Select current batch & make into numpy-int32-format
      current_batch <- d$text_int[c(current_pos:upper)]

      # Make into dict
      feed_dict <- dict(placeholders = current_batch)
      
      # Run current session
      sess$run(fetches = list(train),feed_dict = feed_dict)
      
      # Print
      if(j == 1 | j %% 50 == 0){
        k <- k + 1
        current_cost <- sess$run(loss, feed_dict = feed_dict)
        current_ll <- sess$run(m$log_likelihood, feed_dict = feed_dict)
        cat(paste0('Iteration: ',i,' - Epoch: ',j,'/',epochs,' - Loss: ',current_cost,' - LL: ',current_ll,'\n'))
        loss_stats[k] <- current_cost
        ll_stats[k] <- current_ll
      }
      
      # Update current pos
      current_pos <- upper
      
      # Check if we've processed all data --> reset to 1st pos
      if(current_pos>=length(d$text_int)){current_pos<-1}
    }
  }
  
  # What to return?
  if(args$return_embvectors==TRUE){
    m$output$rho <- sess$run(m$rho)
    m$output$alpha <- sess$run(m$alpha)
  }
  if(args$return_loss==TRUE){
    m$output$loss <- loss_stats
    m$output$ll <- ll_stats
  }
  
  # Close session
  sess$close()
  tf$reset_default_graph()
  return(m)
}




