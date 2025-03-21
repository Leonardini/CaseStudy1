library(readxl)
library(tidyverse)
library(maps)
library(forecast)
library(spdep)  
library(sf)

# Read data from the Excel file from CaseStudy1.xlsx, sheet 'Sheet1'
raw_df <- read_excel('CaseStudy1.xlsx', sheet = 'Sheet1')

names(raw_df) = c("Coordinates","Year","DJF Temp","MAM Temp","JJA Temp","SON Temp", "Population","DJF cases",
              "MAM cases","JJA cases","SON cases")

# Display head for inspection
print(head(raw_df))

names(raw_df)
# Clean and process coordinate column. Remove parentheses, spaces and split into latitude and longitude
clean_df <- raw_df %>% 
  mutate(Coordinates = str_remove_all(`Coordinates`, '\\(') %>% str_remove_all('\\)')) %>%
  separate(Coordinates, into = c('Latitude', 'Longitude'), sep = ',') %>% 
  mutate_all(~{as.numeric(str_trim(.))})



# Create aggregate metrics for each location:
# Total cases (sum of cases from all seasons) and Mean Temperature (average of seasonal temperatures)
summary_df <- clean_df %>%
  rowwise() %>%
  mutate(Total_Cases = sum(c_across(c(`DJF cases`,
                                      `MAM cases`,
                                      `JJA cases`,
                                      `SON cases`)), na.rm = TRUE),
         Mean_Temperature = mean(c_across(c(`DJF Temp`,
                                            `MAM Temp`,
                                            `JJA Temp`,
                                            `SON Temp`)), na.rm = TRUE)) %>%
  ungroup()


clean_df <- clean_df %>%
  rename(n_cases_DJF = `DJF cases`, 
         n_cases_MAM = `MAM cases`, 
         n_cases_JJA = `JJA cases`, 
         n_cases_SON = `SON cases`,
         mean_temp_DJF = `DJF Temp`,
         mean_temp_MAM = `MAM Temp`,
         mean_temp_JJA = `JJA Temp`,
         mean_temp_SON = `SON Temp`)


# Make a copy of the data
cleaned_data <- clean_df


# Reshape the dataset into long format for seasonal analysis
long_data <- cleaned_data %>%
  pivot_longer(cols = c(mean_temp_DJF, mean_temp_MAM, mean_temp_JJA, mean_temp_SON, 
                        n_cases_DJF, n_cases_MAM, n_cases_JJA, n_cases_SON),
               names_to = c('Season', '.value'),
               names_pattern = "(n_cases_|mean_temp_)(.*)")


# The pivot produced two groups: one for temperature (when name starts with season) and one for cases (when starts with n_cases_)
# Separate temperature data from incidence data
temp_df <- cleaned_data %>%
  select(Longitude, Latitude, Year, mean_temp_DJF, mean_temp_MAM, mean_temp_JJA, mean_temp_SON) %>%
  pivot_longer(-c(Year, Longitude, Latitude), names_to = 'Season', values_to = 'Temperature')


cases_df <- cleaned_data %>%
  select(Longitude, Latitude, Year, n_cases_DJF, n_cases_MAM, n_cases_JJA, n_cases_SON) %>%
  pivot_longer(-c(Year, Longitude, Latitude), names_to = 'Season', values_to = 'Cases')


temp_df <- temp_df %>% mutate(Season = case_when(
  Season == 'mean_temp_DJF' ~ 'DJF',
  Season == 'mean_temp_MAM' ~ 'MAM',
  Season == 'mean_temp_JJA' ~ 'JJA',
  Season == 'mean_temp_SON' ~ 'SON'))


cases_df <- cases_df %>% mutate(Season = case_when(
  Season == 'n_cases_DJF' ~ 'DJF',
  Season == 'n_cases_MAM' ~ 'MAM',
  Season == 'n_cases_JJA' ~ 'JJA',
  Season == 'n_cases_SON' ~ 'SON'))


# Merge the temperature and cases dataframe
merged_df <- inner_join(temp_df, cases_df, by = c('Year', 'Season', 'Latitude', 'Longitude'))


# Time-series analysis: Here, we treat each season across the available years as a time series.
seasonal_summary <- merged_df %>%
  group_by(Season, Year) %>%
  summarise(Mean_Temperature = mean(Temperature, na.rm = TRUE),
            Mean_Cases = mean(Cases, na.rm = TRUE),
            Total_Cases = sum(Cases, na.rm = TRUE),
            Regions = n())


print(seasonal_summary)


# Plot the relationship between temperature and cases for each season
p <- ggplot(merged_df, aes(x = Temperature, y = Cases)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = 'lm', se = TRUE, color = 'blue') +
  facet_wrap(~Season) +
  labs(title = 'Relationship between Temperature and Pathogen Cases by Season',
       x = 'Mean Temperature (°C)', y = 'Number of Cases') +
  theme_minimal()


# Save the plot
ggsave('season_temperature_relationship.png', plot = p, width = 10, height = 6)


library(maps)


# Retrieve world map boundaries from maps package
world_map <- map_data('world')


clean_df_aggregate <- clean_df %>%
  mutate(Total_Cases = n_cases_DJF + n_cases_MAM + n_cases_JJA + n_cases_SON) %>%
  select(-starts_with("n_cases")) %>%
  mutate(Annual_Incidence = Total_Cases/Population)


# Create a scatter plot overlaying the data on the world map
map_plot <- ggplot() +
  geom_polygon(data = world_map, aes(x = long, y = lat, group = group), fill = 'gray90', color = 'gray50') +
  geom_point(data = clean_df_aggregate, aes(x = Longitude, y = Latitude, color = Annual_Incidence, size = Population), alpha = 0.8) +
  scale_color_viridis_c() +
  scale_x_continuous(limits = c(-20, 60), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-40, 40), expand = c(0, 0)) +
  labs(title = 'Geographic Distribution of Total Cases by Location',
       x = 'Longitude', y = 'Latitude',
       color = 'Total Cases',
       size = 'Population') +
  theme_minimal()


print(map_plot)


# Save the plot to a file
ggsave('geographic_total_cases_map.png', map_plot, width = 8, height = 6)


# Aggregate incidence by Year
yearly_incidence <- clean_df_aggregate %>%
  group_by(Year) %>% 
  summarise(Avg_Incidence = mean(Annual_Incidence, na.rm = TRUE)) %>% 
  arrange(Year)
print(yearly_incidence)


# Create time series
inc_ts <- ts(yearly_incidence$Avg_Incidence, start = min(yearly_incidence$Year), frequency = 1)
print(inc_ts)


# Fit ARIMA model
fit_model <- auto.arima(inc_ts)
model_summary <- summary(fit_model)
print(model_summary)


# Forecast next 2 years
forecast_model <- forecast(fit_model, h = 2)
print(forecast_model)


# Plot and save forecast
jpeg('incidence_forecast.jpeg', width = 480, height = 480)
plot(forecast_model, main = 'Incidence Forecast over Time')
dev.off()

df_sf <- st_as_sf(cleaned_data, coords = c("Longitude", "Latitude"), crs = 4326)
df_sf_projected <- st_transform(df_sf, crs = 32633)  # adapt as necessary
df_sf_projected$Cases = df_sf_projected$n_cases_DJF + df_sf_projected$n_cases_MAM + 
  df_sf_projected$n_cases_JJA + df_sf_projected$mean_temp_SON
coords <- st_coordinates(df_sf_projected)
knn_5 <- knearneigh(coords, k = 5)
nb_5  <- knn2nb(knn_5)
lw <- nb2listw(nb_5, style = "W")

local_moran <- localmoran(df_sf_projected$Cases, lw)

# The result is a matrix with columns:
#  1. Ii      = local Moran's I statistic
#  2. E.Ii    = expectation of local Moran's I
#  3. Var.Ii  = variance of local Moran's I
#  4. Z.Ii    = standardized I
#  5. Pr(z>0) = p-value for the test

df_sf_projected$Ii    <- local_moran[,1]  # local Moran's I
df_sf_projected$Z.Ii  <- local_moran[,4]  # standard deviate
df_sf_projected$P.Ii  <- local_moran[,5]  # p-value

# Map the local Moran’s I "hotspots" (e.g., high Ii + significant p-value)
# We highlight statistically significant hotspots using P.Ii < 0.05
df_sf_projected$Significant <- ifelse(df_sf_projected$P.Ii < 0.05, "Yes", "No")

g = ggplot() +
  geom_sf(data = df_sf_projected,
          aes(colour = Significant)) +
  labs(title = "Significant Local Moran’s I Hotspots (p < 0.05)",
       fill  = "Significant") +
  theme_minimal()
g
ggsave("moran-hotspots.png",g,width=8,height=6)


#----------------------------------------------------------------------------
# 2. Getis-Ord Gi* Statistic
#    - Another common hotspot analysis method
#----------------------------------------------------------------------------

# localG() from spdep calculates the Getis-Ord Gi* statistic
local_g <- localG(df_sf_projected$Cases, lw)
df_sf_projected$Gi   <- as.numeric(local_g)

# If Gi is a high positive value AND significant, it suggests a hotspot.
g = ggplot() +
  geom_sf(data = df_sf_projected,
          aes(colour = Gi)) +
  scale_fill_viridis_c(option = "plasma") +
  labs(title = "Getis-Ord Gi* Statistic for Cases",
       fill  = "Gi*") +
  theme_minimal()

ggsave("getis-ord.png",g,width=8,height=6)

#############################################################################
## Interpretations:
##  - Local Moran’s I (LISA) can highlight areas surrounded by similar high
##    (hotspots) or low (cold spots) values. The sign of Ii and the significance
##    level (P.Ii) help identify these clusters.
##  - Getis-Ord Gi* focuses on whether each location is part of a cluster of
##    high (hotspots) or low (cold spots) values. Large positive Gi* = hotspot.
#############################################################################


# Display session info with package versions for reproducibility
session_info <- sessionInfo()
print(session_info)


# Save session info to a text file
capture.output(session_info, file = 'session_info.txt')
