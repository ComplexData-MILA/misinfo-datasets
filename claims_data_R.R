# CODES - A GUIDE TO MISINFORMATION DETECTION DATASET

# Librairies --------------------------------------------------------------

library(readr)
library(tidyverse)
library(randomForest)
library(caret)
library(stringr)
library(tidytext)
library(quanteda)
library(jsonlite)
library(xlsx)
library(rdflib)

# Data - Claims --------------------------------------------------------------------

dat_claims <- read_csv("C:/Users/client/Desktop/datasets/R/2024-06-19_dat_claims.csv")

dat_claims %>% 
  select(dataset) %>% 
  table()

# Create veracity variable (1= real, 2= fake, 3= not determined) ----------

dat_claims <- dat_claims %>%
  mutate(veracity = case_when(
    label %in% c("half-true", "covid true", "disputed", "mostly true", "MOSTLY TRUE", 
                 "non conspiracy", "non-covid true", "not misinformation", "real", 
                 "support", "supported", "supports", "SUPPORTS", "true", "TRUE", "correct", "correct attribution", 
                 "CORRECT ATTRIBUTION", "true prevention", "true public health response", "mostly-true", "true but", 
                 "correct", "non-rumor") ~ 1,
    label %in% c("5g covid conspiracy", "barely-true", "conspiracy", "conspiracy, fake remedy",
                 "conspiracy, false reporting", "covid fake", "fake", "Fake", "fake cure",
                 "fake news", "fake remedy", "fake remedy, conspiracy", "fake remedy, false reporting",
                 "fake treatment", "false", "FALSE", "false and misleading", "false fact or prevention",
                 "false public health response", "false reporting", "false reporting, conspiracy",
                 "false reporting, fake remedy", "misinformation", "MISCAPTIONED", "miscaptioned",
                 "misleading", "misleading/false", "misinformation / conspiracy theory", "mostly false",
                 "MOSTLY FALSE", "non-covid fake", "not true", "not_supported", "other conspiracy",
                 "pants on fire", "pants-fire", "refute", "refuted", "refutes", "REFUTES", "scam", "SCAM",
                 "out-of-context", "partly true/misleading", "unlikely") ~ 2,
    label %in% c("ambiguous or hard to classify", "complicated/hard to categorise", "in dispute",
                 "mixed", "mixture", "MIXTURE", "not enough info", "NOT ENOUGH INFO", "not_enough_info",
                 "notenoughinfo", "UNDETERMINED", "(org. doesn't apply rating)", "outdated", 
                 "OUTDATED", "UNPROVEN", "unproven", "no evidence", "half true", "half truth", 
                 "partially correct", "partially false", "partially true", "partly false", 
                 "partly true", "unverified") ~ 3,
    label %in% c("barack-obama", "bill-mccollum", "brian-kemp", "calling out or correction", 
                 "collections", "columbianchemicals", "commercial activity or promotion", 
                 "doug-macginnitie", "emergency", "explanatory", "irrelevant", "irrevelant", 
                 "livr", "news", "Not Applicable", "panic buying", "passport", "pigfish", 
                 "politics", "research in progress", "RESEARCH IN PROGRESS", "two pinocchios", 
                 "CORRECT ATTRIBUTION", "MISATTRIBUTED", "sarcasm or satire", "counter-misinformation", 
                 "labeled satire", "misattributed", "other", "lost legend", "LEGEND", "legend", "suspicions") ~ NA_real_
  ))


# Veracity count - Label Disparity per Dataset ----------------------------

datasets <- unique(dat_claims$dataset)

# List for results
results_table <- list()
results_prop <- list()

# Loop
for (dataset_name in datasets) {
  current_data <- dat_claims %>% 
    filter(dataset == dataset_name) %>% 
    select(where(~ !all(is.na(.))))
  
  result_table <- table(current_data$veracity)
  results_table[[dataset_name]] <- result_table
  
  result_prop <- prop.table(result_table) * 100
  results_prop[[dataset_name]] <- result_prop
}

# Results
results_table
results_prop


# Keywords count ----------------------------------------------------------

stopwords("en")

# Counts + separate analysis by dataset
word_counts_by_dataset_2 <- dat_claims %>%
  filter(!is.na(claim)) %>%
  mutate(claim = str_to_lower(claim)) %>% 
  unnest_tokens(word, claim) %>%
  filter(!word %in% c(stopwords("en"))) %>%
  group_by(dataset) %>%
  count(word) %>%
  filter(n >= 10) %>%
  ungroup()

# true vs false
word_counts_by_label_2 <- dat_claims %>%
  filter(!is.na(claim)) %>%
  mutate(claim = str_to_lower(claim)) %>%
  unnest_tokens(word, claim) %>%
  filter(!word %in% c(stopwords("en"))) %>%
  group_by(dataset, word, veracity) %>%
  summarise(count = sum(!is.na(veracity))) %>%
  ungroup()

# Difference
word_counts_summary_2 <- word_counts_by_label_2 %>%
  pivot_wider(names_from = veracity, 
              values_from = count, 
              values_fill = 0) %>%
  mutate(difference = abs(`1` - `2`)) %>%
  arrange(desc(difference))

# %
word_counts_summary <- word_counts_summary_2 %>%
  rename(real = `1`, false = `2`, other = `3`) %>%
  mutate(sum_obs = real + false)

word_counts_summary <- word_counts_summary %>%
  mutate(pct_real = (real / sum_obs) * 100,
         pct_false = (false / sum_obs) * 100,
         pct_other = (other / sum_obs) * 100)

word_counts_summary <- word_counts_summary %>% 
  select(-c(`NA`))

# Disparity between label 

result_veracity <- dat_claims %>%
  group_by(dataset, veracity) %>%
  summarize(frequency = n(), .groups = 'drop') %>%
  group_by(dataset) %>%
  mutate(percentage = round(frequency / sum(frequency) * 100, 2))

# Rotate result_veracity

## frequency
pivot_01 <- result_veracity %>%
  select(dataset, veracity, frequency) %>%
  pivot_wider(names_from = veracity, values_from = frequency, names_prefix = "frequency_")

## %
pivot_02 <- result_veracity %>%
  select(dataset, veracity, percentage) %>%
  pivot_wider(names_from = veracity, values_from = percentage, names_prefix = "percentage_")

# Combine 2 pivots
result_veracity_pivot <- pivot_01 %>%
  left_join(pivot_02, by = "dataset")

final_data <- word_counts_summary %>%
  left_join(result_veracity_pivot, by = "dataset")

