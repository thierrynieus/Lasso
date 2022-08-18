rm(list = ls())

library(tidyverse) # in spark
library(ggraph) # only for dataset
library(igraph) # in R

temp <- list.files(path = "export_beta/",
                   pattern="*.csv",
                   recursive = TRUE,
                   full.names = FALSE)

filesStruct <- tibble(name = temp) %>%
  separate(col = name,
           into=c("nome",NA),
           sep="\\.csv",
           remove = FALSE) %>%
  separate(col = nome,
           into=c("folder", "nome"),
           sep="/",
           remove = TRUE) %>%
  separate(col = nome,
           into=c("left", "type"),
           sep="_") %>%
  separate(col = left,
           into=c("type2", "value"),
           sep="=") 

filesStruct <- filesStruct %>%
  mutate(folder= as.integer(folder), 
         value = as.factor(value),
         valueD = as.numeric(levels(filesStruct$value))[value],
         across(starts_with("type"),as.factor))

AllData <- filesStruct %>%
  filter(type %in% c("exc","inh")) %>%
  pull(name) %>%
  lapply(X = ., FUN = function(name){
  name_file <- paste0("export_beta/",name)
  return(read_delim(file = name_file,delim = "\t",
                    col_types = c("ii"),
                    show_col_types = FALSE) %>%
           mutate(name = name))
  }) %>%
  bind_rows() %>%
  left_join(filesStruct)
  
Truestr <- filesStruct %>%
  filter(type %in% c("connmat")) %>%
  pull(name) %>%
  lapply(X = ., FUN = function(name){
    name_file <- paste0("export_beta/",name)
    return(read_delim(file = name_file,delim = "\t",
                      col_types = c("ii"),
                      show_col_types = FALSE) %>%
             mutate(name = name,
                    real = TRUE))
  }) %>%
  bind_rows() %>%
  left_join(filesStruct %>%
              filter(type %in% c("connmat")) %>%
              select(name,folder)) %>%
  select(-name)

AllData <- filesStruct %>%
  filter(type %in% c("exc","inh")) %>%
  pull(name) %>%
  lapply(X = ., FUN = function(name){
    name_file <- paste0("export_beta/",name)
    return(read_delim(file = name_file,delim = "\t",
                      col_types = c("ii"),
                      show_col_types = FALSE) %>%
             mutate(name = name))
  }) %>%
  bind_rows() %>%
  left_join(filesStruct) %>%
  left_join(Truestr, by = c("folder","from","to")) %>%
  replace_na(list(real = FALSE))

save(file = "dataset.RDATA",list = "AllData")


distGraphs <- lapply(Truestr %>% 
                       distinct(folder) %>%
                       pull,function(nfold) {
                         gloc <- graph_from_data_frame(directed=TRUE,
                                                       d = Truestr %>%
                                                         filter(folder == nfold) %>%
                                                         select(from,to))
                         return(distances(
                           gloc,
                           v = V(gloc),
                           to = V(gloc),
                           mode = c("out")))
                       })

tableFreqDistances <- lapply(seq_len(length(distGraphs)),
                             function(n) {
                               dists <- distGraphs[[n]]
                               return(as.data.frame(table(dists)) %>%
                                        as_tibble() %>%
                                        mutate(folder = n))
                             }) %>%
  bind_rows()

AllDataWithDistances <- AllData %>%
  rowwise() %>%
  mutate(dists = distGraphs[[folder]][from+1L,to+1L]) %>%
  ungroup() %>%
  mutate(dists = as_factor(dists))


AnalysisDistances <- AllDataWithDistances %>%
  group_by(folder,dists,value) %>%
  summarise(nCounts = n()) %>% 
  full_join(tableFreqDistances, by = c("folder","dists"))

save(file = "datasetWdist.RDATA",
     list = c("AllDataWithDistances",
              "tableFreqDistances",
              "AnalysisDistances"))

p1 <- ggplot(data = AnalysisDistances %>% filter(dists != "1") %>%
               mutate(lambda = as.numeric(levels(AnalysisDistances$value))[value],
                      percIn = nCounts/Freq),
             mapping = aes(x = 1/lambda, y = percIn,
                           color = dists)) + 
  scale_y_continuous(labels = scales::percent) +
  geom_line() +
  facet_wrap( ~ folder, ncol=2) +
  theme_bw()

p2 <- ggplot(data = AnalysisDistances %>% # filter(dists != "1") %>%
               mutate(lambda = as.numeric(levels(AnalysisDistances$value))[value],
                      percIn = nCounts/Freq),
             mapping = aes(x = 1/lambda, y = percIn,
                           color = dists)) + 
  scale_y_continuous(labels = scales::percent) +
  geom_line() +
  facet_wrap( ~ folder, ncol=2) +
  theme_bw()

ggsave('ByFolder_no1.png',p1,width=7.3,height=5.2)
ggsave('ByFolder.png',p2,width=7.3,height=5.2)
