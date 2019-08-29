# -----------------------------------------------------------------
#
#
#       Interpretable Word Embeddings via Informative Priors
#                           >> Data >>
#
#
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# Function for basic string-cleaning
# -----------------------------------------------------------------
basic_cleaning <- function(mystring){
  mystring <- stringi::stri_replace_all(str = mystring, regex = '[:digit:]', replacement = 'x')
  mystring <- stringi::stri_replace_all(str = mystring, regex = "[^[:alnum:]]", replacement = ' ')
  mystring <- tolower(mystring)
  mystring = stringr::str_replace(gsub("\\s+", " ", stringr::str_trim(mystring)), "B", "b")
  mystring = gsub("^\\s+|\\s+$", "", mystring)
  mystring <- stringr::str_replace_all(mystring," "," ")
  return(mystring)
}
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# Prepare data for word embedding estimation
# -----------------------------------------------------------------
process_text_string <- function(text_string, 
                                vocab_size = 1000){
  
  # Text as data.table with rows representing tokens
  long_dt <- data.table(w=strsplit(x = text_string, split = ' ', perl = T)[[1]])
  long_dt[,order_id := .I]
  
  # Get the frequency of each unique word, and only keep top 'vocab_size' words
  word_count_dt <- long_dt[,.(count=.N),by='w']
  word_count_dt <- word_count_dt[order(count,decreasing = T)][1:min(c(vocab_size,nrow(word_count_dt))),]
  
  # Create numeric id for each unique word
  word_count_dt[,wid := 0:(.N-1)]
  
  # Dictionary translating numberic/character term-id
  dictionary <- word_count_dt[,.(w,wid)]
  
  # Replace character with numeric in data
  long_dt <- merge(x=long_dt,
                   y=dictionary,
                   by='w')
  long_dt <- long_dt[order(order_id,decreasing = F)]
  long_dt[,order_id := NULL]
  
  # Unigram
  unigram <- word_count_dt[,.(w, wid,count,prop=count/sum(count))]
  unigram[,sampling_value := prop^(3/4)]
  unigram <- unigram[,.(w, wid, count, sampling_value = sampling_value/sum(sampling_value))] # re-normalize sampling_value
  
  # Return
  return(list(text_int   = long_dt$wid,
              dictionary = dictionary,
              Ntokens    = nrow(long_dt),
              unigram    = unigram))
  
}
# -----------------------------------------------------------------


# -----------------------------------------------------------------
# Download a sample of Gutenberg books directly to R
# -----------------------------------------------------------------
dl_sample_gutenberg <- function(n = 20,
                                category = 'Bestsellers, American, 1895-1923'){
  
  # Load (& if needed install) package for dl'ing data
  library(gutenbergr)
  
  # DL sample of Gutenberg books
  works <- as.data.table(gutenberg_works())
  # - specify category
  if(!is.null(category)){
    works_sample_id <- works[has_text==TRUE][gutenberg_bookshelf==category]
  }
  # - sample
  if(n > nrow(works_sample_id)){
    works_sample_id <- works_sample_id[sample(1:.N,n),]
  }
  works_sample <- lapply(works_sample_id$gutenberg_id,function(x)as.data.table(gutenberg_download(x,strip = T)))
  
  # Collapse text within books
  works_sample <- lapply(works_sample, function(x) paste(x$text, collapse = ' '))
  
  # Collapse text across books
  works_sample <- paste(unlist(works_sample),collapse = ' ')
  
  # Return
  return(works_sample)
}
# -----------------------------------------------------------------
