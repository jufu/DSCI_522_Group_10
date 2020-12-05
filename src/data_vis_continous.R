#author: Asma Al-Odaini
#date: 2020-11-25
"This script imports the training data file and generates visualizations to explore the numerical features.

Usage: data_vis_continous.R --data_path=<data_path> --image_path=<image_path>

Options:
--data_path=<data_path>      Takes a path of the training data csv
--image_path=<image_path>     Takes a path for images

Example:
  Rscript src/data_vis_continous.R --data_path='data/processed/bank-additional-full_train.csv' --image_path='results/'
" -> doc


library(tidyverse, quietly = T)
library(ggridges)
library(ggthemes)
library(docopt)

opt <- docopt(doc)


main <- function(opt) {
  train_df <- read_csv(opt$data_path, col_types = cols())
  train_df <- read_csv('data/processed/bank-additional-full_train.csv')
  train_df <- train_df %>% 
    mutate(y = case_when(target == "1" ~ "Purchased",
                         target == "0"  ~ "Not Purchased"))

  train_df <- train_df %>%
    mutate(target = as.factor(target))
  train_df

  train_df <- train_df %>% select(age, last_contact_duration :previous_contacts, employment_variation_rate : target)
  train_df

  train_df <- train_df %>% rename("Age" = "age",
                                  "Last_Contact_Duration" = "last_contact_duration",
                                  "Number_of_Contacts_During_Campaign" = "contacts_during_campaign",
                                  "Days_After_Previous_Contact" = "days_after_previous_contact",
                                  "Number_of_Previous_Contacts" = "previous_contacts",
                                  "Employment_Variation_Rate" = "employment_variation_rate",
                                  "Consumer_Price_Index" = "consumer_price_index",
                                  "Consumer_Confidence_Index" = "consumer_confidence_index",
                                  "Euribor_3_Month_Rate" = "euribor_3_month_rate",
                                  "Number_of_Employees" = "number_of_employees")
  train_df

  grid <- train_df %>%
    gather(key = feature, value = value, -target) %>%
    mutate(feature = str_replace_all(feature, "_", " ")) %>%
    ggplot(aes(x = value, y = target, colour = target, fill = target)) +
    facet_wrap(. ~ feature, scale = "free", ncol = 2) +
    geom_density_ridges(alpha = 0.8) +
    scale_fill_tableau() +
    scale_colour_tableau() +
    guides(fill = FALSE, color = FALSE) +
    theme(axis.title.x = element_blank(),
          axis.title.y = element_blank())
  path <- paste(opt$image_path, 'numeric.png',sep="")
  ggsave(path, grid)
  print("A density plot was created and saved in the declared image path")

  }

main(opt)



