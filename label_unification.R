###############################
## VERACITY VARIABLE - LABEL ##
##        2025-01-26         ##
###############################

# LIBRARIES

library(readr)
library(tidyverse)

# DATA

dat_claims <- read_csv("C:/Users/client/Desktop/datasets/2025_01_dat_claims.csv")

dat_claims %>% 
  select(dataset) %>% 
  table()

# Create veracity variable ----------

## 1- True
## 2- False
## 3- Mixed
## 9- Unknown (including NAs, unrelated and unproven)

dat_claims <- dat_claims %>% 
  mutate(veracity = case_when(
    label %in% c("covid true", "mostly true", "MOSTLY TRUE", "Correct Attribution",
                 "non conspiracy", "non-covid true", "not misinformation", "real", 
                 "support", "supported", "supports", "SUPPORTS", "true", "TRUE", "correct", "correct attribution", 
                 "CORRECT ATTRIBUTION", "true prevention", "true public health response", "mostly-true", "true but", 
                 "correct", "non-rumor", "counter-misinformation", "Mostly true", "Mostly True", "True",
                 "Correct", "True but", "truth", "correctly attributed", "mostly correct", "true but outdated",
                 "truth but not the one you think","was true, but the promotion is now over", "legit",
                 "true, but the boycott has ended", "mostly-correct", "barely-true", "was true; now outdated", "was true, but the boy has since passed away",
                 "mostly truth", "truth but an opinion", "previously truth &amp; now resolved", "truth but it is an opinion",
                 "reported to be true", "was true, but the program has since ended", "truth but a conservation effort", "real photograph", 
                 "truth but resolved", "truth & outdated", "was true", "was true, but has now ended", "previously truth & now resolved",
                 "previously truth but now resolved", "was true; fulfillment level has now been reached") ~ 1,
    label %in% c("5g covid conspiracy", "conspiracy", "conspiracy, fake remedy",
                 "conspiracy, false reporting", "covid fake", "fake", "Fake", "fake cure",
                 "fake news", "fake remedy", "fake remedy, conspiracy", "fake remedy, false reporting",
                 "fake treatment", "false", "FALSE", "false and misleading", "false fact or prevention",
                 "false public health response", "false reporting", "false reporting, conspiracy",
                 "false reporting, fake remedy", "misinformation", "MISCAPTIONED", "miscaptioned",
                 "misleading", "misleading/false", "misinformation / conspiracy theory", "mostly false",
                 "MOSTLY FALSE", "non-covid fake", "not true", "not_supported", "other conspiracy",
                 "pants on fire", "pants-fire", "refute", "refuted", "refutes", "REFUTES", "scam", "SCAM",
                 "out-of-context", "partly true/misleading", "unlikely", "False information", "Altered photo", 
                 "False information and graphic content", "False, Partly false information", 
                 "Misinformation / Conspiracy theory", "PANTS ON FIRE", "Scam", "MISATTRIBUTED", "misattributed",
                 "two pinocchios", "sarcasm or satire", "labeled satire", "lost legend", "LEGEND", "legend",  "Labeled Satire", 
                 "False", "Misleading", "MISLEADING", "Mostly false", "Misleading/False", "Misattributed", "Mostly False",
                 "Miscaptioned", "Not true", "mislEADING", "Two Pinocchios", "MiSLEADING", "Fake news", "reported as fiction",
                 "decontextualized", "incorrect", "altered image", "altered photo", "fiction", "altered", "altered video",
                 "understatedexaggerated", "satire", "incorrectly", "false, altered news graphic", "hoax", "incorrect attribution",
                 "unsubstantiated", "probably false", "mostly fiction", "falso", "satiremanipulated", "faux", "fabricated article",
                 "doctored image", "this altered photo does not show the arrest of ethiopia’s former head of intelligence", "inaccurate",
                 "fal", "manipulated image", "falsethis image shows the memorial service for lol mahamat choua, chad’s fourth president who died in 2019",
                 "photo détournée", "fiction &amp; satire", "False", "Altered video", "False information, Partly false information",
                 "Altered Photo/Video", "False, False information", "False information, Partly false information.", "Altered photo/video.",
                 'False information, False information.', "False information, False information and graphic content", "False headline", 
                 "False information, Missing context", "False information., Partly false information", "False information, Missing Context",
                 "Altered photo/video", "False information, Partly False", "inaccurate attribution", "incorrectly attributed", "misleading 3.5",
                 "digital manipulations", "fiction & satire", "satiremanipulated image", "may be misleading", "photo out of context", "fasle",
                 "False information.", "calling out or correction", "originated as satire") ~ 2,
    label %in% c("in dispute", "mixed", "mixture", "MIXTURE", "half true", "half truth", 
                 "partially correct", "partially false", "partially true", "partly false", "partly true", 
                 "Partly false information", "IN DISPUTE", "suspicions", "Suspicions", "PARTLY FALSE", "Mixture",
                 "Partly False", "Half True", "PARTLY TRUE", "Partially false", "Partly true", "Partly FALSE",
                 "HALF TRUTH", "HALF TRUE", "Partially correct", "Mixed", "Partially true", "half false", "partlyfalse",
                 "mixture of true and false information", "real photographs; inaccurate description", "sort of", "half-flip",
                 "truth &amp; misleading", "was true, but her conviction has been overturned", "truth &amp; fiction",
                 "truth &amp; outdated", "real photograph; inaccurate description", "truth &amp; unproven",
                 "mixture of truth and falsity", "not quite", "mixture of true and outdated information",
                 "truth but obama quote is fiction", "real picture; inaccurate description", "truth &amp; fiction &amp; disputed",
                 "truth but misleading", "possible, but not common", "real photos; inaccurate description", "truth, fiction, and unproven",
                 "maybe", "truth fiction &amp; disputed", "real photo; inaccurate description", "Partly false information.",
                 "Partly False, Partly false information", "Partly false information, Partly false information.", 
                 "Missing context, Partly false information", "Missing context., Partly false information", "Partly false", "truth & misleading",
                 "truth & fiction", "real photograph;  inaccurate description", "truth & unproven", "real photographs;  inaccurate description",
                 "mixture of true and  outdated information", "fiction & disputed", "truth fiction & unproven", "truth & fiction & disputed",
                 "real video; inaccurate description", "reported as truth & disputed", "was true, but isnt any more", "truth fiction & disputed",
                 "truth & disputed", "half-true", "disputed") ~ 3,
    TRUE ~ 9
  ))


## Number of observations per label
table <- dat_claims %>% 
  select(veracity) %>% 
  table()
table

percentages <- prop.table(table) * 100
percentages

## Unique label values when veracity == 9
count_labels_9 <- dat_claims %>% 
  filter(veracity == 9) %>% 
  group_by(label) %>% 
  summarise(n = n(), .groups = "drop") %>% 
  arrange(desc(n))

count_labels_9

## Write .csv
write.csv(dat_claims, "2025_01_26_dat_claims.csv", row.names = FALSE)
