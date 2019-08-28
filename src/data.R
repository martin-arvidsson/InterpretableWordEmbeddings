# -----------------------------------------------------------------
#
#
#       Interpretable Word Embeddings via Informative Priors
#                           >> Data >>
#
#
# -----------------------------------------------------------------



# -----------------------------------------------------------------
# Import/Export.txt files
# -----------------------------------------------------------------
# - Import
import_txt_file <- function(file){
  con <- file(file, open='r')
  x <- readLines(con)
  close(con)
  return(x)
}
# - Export
export_txt_file <- function(object,file){
  con <- file(file)
  writeLines(object, con)
  close(con)
}
# -----------------------------------------------------------------



# -----------------------------------------------------------------
# Create 'unigram_dt'
# -----------------------------------------------------------------
process_text_string <- function(text_string, 
                                vocab_size = 1000, 
                                sample_size = NULL){
  
  # # Test
  # text_string = mystring
  # vocab_size = 500
  # sample_size = NULL
  
  # Derive long data.table from text
  long_dt <- data.table(w=strsplit(x = text_string, split = ' ', perl = T)[[1]])
  
  # Create order variable
  long_dt[,order_id := .I]
  
  # Sample?
  if(!is.null(sample_size)){
    if(sample_size>nrow(long_dt)){
      cat('Requested sample size greater than the size of the data. Continues with full dataset')
    }else{
      start_id <- sample(1:(nrow(long_dt)-sample_size-1),size = 1)
      long_dt <- long_dt[start_id:(start_id+sample_size)]
      long_dt[,order_id := .I]
    }
  }
  
  # Get the frequency of each unique word, and keep top 'vocab_size' words
  word_count_dt <- long_dt[,.(count=.N),by='w']
  word_count_dt <- word_count_dt[order(count,decreasing = T)][1:min(c(vocab_size,nrow(word_count_dt))),]
  word_count_dt[,wid := 0:(.N-1)]
  
  # Create numeric id for each unique word (among the top L ones)
  dictionary <- word_count_dt[,.(w,wid)]
  
  # Replace words with word-ids in data
  long_dt <- merge(x=long_dt,y=dictionary,by='w')
  long_dt <- long_dt[order(order_id,decreasing = F)]
  long_dt[,order_id := NULL]
  
  # Text string (but with words replaced with integer ids)
  text_int <- long_dt$wid
  
  # Create "N" variable -- e.g. "long" our data is
  Ntokens <- nrow(long_dt)
  
  # Unigram
  unigram <- word_count_dt[,.(w, wid, count=count, prop = count/sum(count))]
  unigram[,sampling_value := prop^(3/4)]
  unigram <- unigram[,.(w, wid, count, sampling_value = sampling_value/sum(sampling_value))] # re-normalize sampling_value
  
  # Return
  return(list(text_int=text_int,
              dictionary=dictionary,
              Ntokens=Ntokens,
              unigram=unigram))
  
}
# -----------------------------------------------------------------




# -----------------------------------------------------------------
# Function for basic cleaning
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



# Construct data objects
# -----------------------------------------------------------------
construct_data <- function(mystring, min_w_freq = 10, max_terms = 5000){
  
  # Text as data.table with rows representing tokens
  text_dt <- data.table(w=strsplit(x = mystring, split = ' ', perl = T)[[1]])
  text_dt[,original_order := .I]
  
  # Unigram data.table
  unigram <- text_dt[,.(count=.N),by=w][order(count,decreasing = T)]
  unigram <- unigram[,wid := 0:(.N-1)]
  
  # Remove terms with too low frequency?
  if(!is.null(min_w_freq)){
    unigram <- unigram[count>=min_w_freq]
  }
  # Only keep a maximum number of terms?
  if(!is.null(max_terms)){
    unigram <- unigram[1:min(c(nrow(unigram),max_terms))]
  }
  
  # Update text_dt according to removals
  text_dt <- text_dt[w %in% unigram$w]
  
  # Replace character-tokens in text_dt with numeric-tokens in unigram
  text_dt <- merge(x=text_dt,
                   y=unigram[,.(w,wid)],
                   by=c('w'))
  text_dt <- text_dt[order(original_order,decreasing = F)]
  
  # Create list object storing relevant data-info
  d <- list(text_int = text_dt$wid, 
            Ntokens = length(text_dt$wid))
  
  # Set sampling constant for each word
  unigram <- unigram[,.(w, wid, count, prop = count/sum(count))]
  unigram[,sampling_value := prop^(3/4)]
  unigram <- unigram[,.(w, wid, count, sampling_value = sampling_value/sum(sampling_value))] # re-normalize sampling_value
  d$unigram <- unigram
  
  # Dictionary
  d$dictionary <- unigram[,.(w,wid)]
  
  # GC
  gc()
  
  # Return
  return(d)
}
# -----------------------------------------------------------------



# -----------------------------------------------------------------
# Function to import processed-data
# -----------------------------------------------------------------
import_processed_data <- function(fpath, 
                                  reduced_V = NULL, 
                                  reduced_Ntokens = NULL){
  
  ## Test
  ##nmpy_source <- paste0(data_folder0,'efe_priors/data/dat/wiki1_0327/train/')
  ##unigram_source <- paste0(data_folder0,'/efe_priors/data/dat/wiki1_0327/unigram.txt')
  #fpath <- 'C:/Users/arvid/Dropbox/EFE/efe_priors/data/dat/wiki1_0327/'
  #reduced_V <- 10000
  
  # Import & collapse numpy files
  nmpy_files <- list.files(paste0(fpath,'/train/'))
  system.time(nmpy_text <- lapply(nmpy_files,function(x)np$load(paste0(fpath,'/train/',x))))
  nmpy_text <- unlist(nmpy_text)
  
  # Import unigram
  unigram <- fread(input = paste0(fpath,'unigram.txt'), col.names = c('w','wid','count'))
  
  # Reduce size of vocabulary?
  if(!is.null(reduced_V)){
    unigram <- unigram[wid %in% c(0:(reduced_V-1)),]
    nmpy_text <- nmpy_text[nmpy_text %in% unigram$wid]
  }
  
  # Select a subset of tokens?
  if(!is.null(reduced_Ntokens)){
    nmpy_text <- nmpy_text[1:reduced_Ntokens]
  }
  
  # Create list object storing relevant data-info
  d <- list(text_int = nmpy_text, 
            Ntokens = length(nmpy_text))
  
  # Set sampling constant for each word
  unigram <- unigram[,.(w, wid, count, prop = count/sum(count))]
  unigram[,sampling_value := prop^(3/4)]
  unigram <- unigram[,.(w, wid, count, sampling_value = sampling_value/sum(sampling_value))] # re-normalize sampling_value
  d$unigram <- unigram
  
  # Dictionary
  d$dictionary <- unigram[,.(w,wid)]
  
  # GC
  gc()
  
  # Return
  return(d)
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
