#########################################
# Project: Mila Datasets                #
# Coded by: Gabrielle Péloquin-Skulski  #
# Date last edited: July 30, 2024.      # 
#########################################

rm(list=ls())

# Set working directory
if(Sys.info()["user"] == "gabriellepeloquinskulski"){
  setwd("~/Dropbox (MIT)/mila_datasets")
}else{
  setwd("")
}

# Packages
library(dplyr)
library(readr)


dat <- read.csv("partisan_lean.csv") |>
  filter((veracity == 1 | veracity == 2) & 
           (X0 %in% c("A", "B", "C", "D"))  & 
           !startsWith(source_dataset, "1864.json" ) & 
           !source_dataset=="{}" & 
           !source_dataset=="fakecovid") |>
  mutate(party_lean = X0) |>
  select(source_dataset, claim, veracity, party_lean)

true <- dat |> 
  filter(veracity==1)

false <- dat |> 
  filter(veracity==2)

# Create Table that shows percent of true headlines

# Calculate percentages for XO by source_dataset and veracity
result_true <- true |>
  group_by(source_dataset) |>
  summarise(
    total = n(),
    party_lean_A_count = sum(party_lean == "A", na.rm = TRUE),
    party_lean_B_count = sum(party_lean == "B", na.rm = TRUE), 
    party_lean_C_count = sum(party_lean == "C", na.rm = TRUE),
    party_lean_D_count = sum(party_lean == "D", na.rm = TRUE)
  ) |>
  mutate(
    party_lean_A_percent = round((party_lean_A_count / total) * 100, 2),
    party_lean_B_percent = round((party_lean_B_count / total) * 100, 2),
    party_lean_C_percent = round((party_lean_C_count / total) * 100, 2),
    party_lean_D_percent = round((party_lean_D_count / total) * 100, 2)
  ) |> 
  select(source_dataset, party_lean_A_percent, party_lean_B_percent, party_lean_C_percent, party_lean_D_percent)

result_true_all <- true |>
  summarise(
    total = n(),
    party_lean_A_count = sum(party_lean == "A", na.rm = TRUE),
    party_lean_B_count = sum(party_lean == "B", na.rm = TRUE), 
    party_lean_C_count = sum(party_lean == "C", na.rm = TRUE),
    party_lean_D_count = sum(party_lean == "D", na.rm = TRUE)
  ) |>
  mutate(
    party_lean_A_percent = round((party_lean_A_count / total) * 100, 2),
    party_lean_B_percent = round((party_lean_B_count / total) * 100, 2),
    party_lean_C_percent = round((party_lean_C_count / total) * 100, 2),
    party_lean_D_percent = round((party_lean_D_count / total) * 100, 2)
  ) |> 
  select(party_lean_A_percent, party_lean_B_percent, party_lean_C_percent, party_lean_D_percent)

  
result_false <- false |>
  group_by(source_dataset) |>
  summarise(
    total = n(),
    party_lean_A_count = sum(party_lean == "A", na.rm = TRUE),
    party_lean_B_count = sum(party_lean == "B", na.rm = TRUE), 
    party_lean_C_count = sum(party_lean == "C", na.rm = TRUE),
    party_lean_D_count = sum(party_lean == "D", na.rm = TRUE)
  ) |>
  mutate(
    party_lean_A_percent = round((party_lean_A_count / total) * 100, 2),
    party_lean_B_percent = round((party_lean_B_count / total) * 100, 2),
    party_lean_C_percent = round((party_lean_C_count / total) * 100, 2),
    party_lean_D_percent = round((party_lean_D_count / total) * 100, 2)
  ) |>
  select(source_dataset, party_lean_A_percent, party_lean_B_percent, party_lean_C_percent, party_lean_D_percent)

result_false_all <- false |>
  summarise(
    total = n(),
    party_lean_A_count = sum(party_lean == "A", na.rm = TRUE),
    party_lean_B_count = sum(party_lean == "B", na.rm = TRUE), 
    party_lean_C_count = sum(party_lean == "C", na.rm = TRUE),
    party_lean_D_count = sum(party_lean == "D", na.rm = TRUE)
  ) |>
  mutate(
    party_lean_A_percent = round((party_lean_A_count / total) * 100, 2),
    party_lean_B_percent = round((party_lean_B_count / total) * 100, 2),
    party_lean_C_percent = round((party_lean_C_count / total) * 100, 2),
    party_lean_D_percent = round((party_lean_D_count / total) * 100, 2)
  ) |>
  select(party_lean_A_percent, party_lean_B_percent, party_lean_C_percent, party_lean_D_percent)


# Export result_true to a CSV file
write.csv(result_true, "result_true.csv", row.names = FALSE)
write.csv(result_false, "result_false.csv", row.names = FALSE)



