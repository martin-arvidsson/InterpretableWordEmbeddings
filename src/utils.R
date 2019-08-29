# -----------------------------------------------------------------
#
#
#       Interpretable Word Embeddings via Informative Priors
#                           >> Utils >>
#
#
# -----------------------------------------------------------------


options(tensorflow.extract.one_based = FALSE)


# -----------------------------------------------------------------
# Reset data-list by re-shuffling word indices for anchored words
# -----------------------------------------------------------------
reset_d_based_on_anchored_priors <- function(args, d, prior_type=1){
  
  # Identify old indices for words we want to anchor
  args <- identify_prior_idx(args = args, d = d)
  
  # Create data.table of "old-ids" for anchored words
  if(length(args$prior_idx$pos_idx)>0){
    dt1 <- data.table(old_id=args$prior_idx$pos_idx,type='d_pos_idx')
    dt2 <- data.table(old_id=args$prior_idx$neg_idx,type='d_neg_idx')
    dt12 <- rbindlist(list(dt1,dt2),use.names = T, fill = T)
  }else{dt12 <- NULL}
  if(prior_type>=3 & length(args$prior_idx$neutral_idx)>0){
    dt3 <- data.table(old_id=args$prior_idx$neutral_idx,type='n_idx')
    dt123 <- rbindlist(list(dt12,dt3), use.names = T, fill = T)
  }else{
    dt123 <- dt12
  }
  
  # - new ids for prior words
  prior_idx_dt <- unique(dt123,by='old_id')
  prior_idx_dt[,new_id := 0:(.N-1)]
  
  # - new ids for regular words
  rest_idx_dt <- d$unigram[,.(old_id=wid,type='rest')][!old_id %in% unique(prior_idx_dt$old_id)]
  new_rest_ids <- (max(prior_idx_dt$new_id)+1):(nrow(d$unigram)-1)
  rest_idx_dt[,new_id := new_rest_ids]
  
  # - update text according to new ids
  text_dt <- data.table(w=d$text_int)
  text_dt[,order:=.I]
  text_dt <- merge(x=text_dt,
                   y=prior_idx_dt[,.(old_id,new_id_prior=new_id)],
                   by.x='w',by.y='old_id',all.x=T,all.y=F)
  text_dt <- merge(x=text_dt,
                   y=rest_idx_dt[,.(old_id,new_id_regular=new_id)],
                   by.x='w',by.y='old_id',all.x=T,all.y=F)
  text_dt <- text_dt[order(order,decreasing = F)]
  text_dt[!is.na(new_id_prior),new_w := new_id_prior]
  text_dt[is.na(new_id_prior),new_w := new_id_regular]
  
  # - create translator between old and new
  d$old_id_new_id_translator <- unique(text_dt[,.(wid_old=w,wid_new=new_w)])
  d$old_id_new_id_translator <- d$old_id_new_id_translator[order(wid_new,decreasing = F)]
  vocab <- 0:(args$L-1)
  old_not_in_text <- vocab[!vocab %in% d$old_id_new_id_translator$wid_old]
  new_slots_open <- vocab[!vocab %in% d$old_id_new_id_translator$wid_new]
  d$old_id_new_id_translator <- rbindlist(list(d$old_id_new_id_translator,data.table(wid_old=old_not_in_text,wid_new=new_slots_open)))
  d$old_id_new_id_translator <- d$old_id_new_id_translator[order(wid_old)]
  
  # - update unigram
  d$unigram <- merge(x=d$unigram,y=d$old_id_new_id_translator,by.x='wid',by.y='wid_old')
  setnames(d$unigram,c('wid','wid_new'),c('wid_pre_mod','wid'))
  d$unigram <- d$unigram[order(wid,decreasing = F)]
  
  # - update dictionary
  d$dictionary <- d$unigram[,.(wid,w)]
  
  # - replace d$text_int
  text_dt <- text_dt[order(order,decreasing = F)]
  d$text_int <- text_dt$new_w
  
  # - insert vector of anchored idx
  d$prior_idx_dt <- prior_idx_dt
  
  # Return
  return(list(args=args,d=d))
}
# -----------------------------------------------------------------




# -----------------------------------------------------------------
# Prepare anchored embeddings
# -----------------------------------------------------------------

# (1)
identify_prior_idx <- function(args, d){
  args$prior_idx$pos_idx = d$dictionary[w %in% args$prior_list$categ1,]$wid
  args$prior_idx$neg_idx = d$dictionary[w %in% args$prior_list$categ2,]$wid
  args$prior_idx$neutral_idx = d$dictionary[w %in% args$prior_list$categ3,]$wid
  args$prior_idx$all_special_idx = unique(c(args$prior_idx$pos_idx, 
                                            args$prior_idx$neg_idx, 
                                            args$prior_idx$neutral_idx))
  args$prior_idx$regular_idx = d$dictionary$wid[!d$dictionary$wid %in% args$prior_idx$all_special_idx]
  return(args)
}

# (2)
anchored_prior_prelims <- function(m, d, args){
  
  # (i) Select special dim(s)
  m$sdim0 <- args$K - 1          # for code tat applies to Python/TF objects (where indices begin at 0)
  m$sdim1 <- args$K              # for code that applies to R objects (where indices begin at 1)
  
  # (ii) Identify word index for considered words
  if(args$prior_list$prior_type > 0){
    m$pos_idx = d$dictionary[w %in% args$prior_list$categ1,]$wid
    m$neg_idx = d$dictionary[w %in% args$prior_list$categ2,]$wid
    m$pos_idx1 <- m$pos_idx + 1
    m$neg_idx1 <- m$neg_idx + 1
  }
  if(args$prior_list$prior_type >= 3){
    m$neutral_idx = d$dictionary[w %in% args$prior_list$categ3,]$wid
    m$neutral_idx1 <- m$neutral_idx + 1
  }
  
  # (iii) Index for remaining / not-considered words
  m$all_special_idx = unique(c(m$pos_idx, m$neg_idx, m$neutral_idx))
  m$regular_idx = d$dictionary$wid[!d$dictionary$wid %in% m$all_special_idx]
  
  # (iv) Modify "initializing vectors" for special words
  # - rho -
  if('rho' %in% args$prior_list[['vectors']]){
    if(args$prior_list[['prior_type']]>=1){
      m$rho_init[m$pos_idx1, m$sdim1]       <- 1
      m$rho_init[m$neg_idx1, m$sdim1]       <- -1
      m$rho_init[m$pos_idx1, 1:(m$sdim1-1)] <- 0
      m$rho_init[m$neg_idx1, 1:(m$sdim1-1)] <- 0
    }
    if(args$prior_list[['prior_type']] >= 3){
      m$rho_init[m$neutral_idx1, m$sdim1] <- 0
    }
  }
  # - alpha -
  if('alpha' %in% args$prior_list[['vectors']]){
    if(args$prior_list[['prior_type']] >=1){
      m$alpha_init[m$pos_idx1, m$sdim1]       <- 1
      m$alpha_init[m$neg_idx1, m$sdim1]       <- -1
      m$alpha_init[m$pos_idx1, 1:(m$sdim1-1)] <- 0
      m$alpha_init[m$neg_idx1, 1:(m$sdim1-1)] <- 0
    }
    if(args$prior_list[['prior_type']] >= 3){
      m$alpha_init[m$neutral_idx1, m$sdim1] <- 0
    }
  }
  
  # (iv) Return
  return(m)
}
# -----------------------------------------------------------------



