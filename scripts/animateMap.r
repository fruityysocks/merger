# Load necessary libraries
library(maps)
library(ggplot2)
library(gganimate)
library(dplyr)

# Load your data
x <- read.csv("data/demographicInfo.csv")

# Create a list to store all data points for each day
all_points <- list()

# Loop through each day and create a data frame with all data points for that day
for (day in 1:3) {
  day_points <- data.frame(
    childLat = x[, paste0("childLatJittered_day", day)],
    childLong = x[, paste0("childLongJittered_day", day)],
    day = day
  )
  
  # Add the data frame to the list
  all_points[[day]] <- day_points
}

# Combine all points into a single data frame
combined_points <- do.call(rbind, all_points)

# Create a map with state boundaries
new_england <- map_data("state") %>%
  filter(region %in% c("maine", "new hampshire", "vermont", "massachusetts", "rhode island", "connecticut"))

# Create the plot
p <- ggplot() +
  geom_polygon(data = new_england, 
               aes(x = long, y = lat, group = group), 
               color = "black", fill = "lightgrey") +
  geom_point(data = combined_points, 
             aes(x = childLong, y = childLat, color = factor(day)), 
             size = 2, alpha = 0.7) +
  coord_fixed(xlim = c(-73.5, -65), ylim = c(41.15, 48)) +
  theme_void() +
  labs(title = "Movement", color = "Day") +
  scale_color_manual(values = c("1" = "blue", "2" = "blue", "3" = "blue"))

# Create and save the animation
anim <- p + 
  transition_states(day, transition_length = 2, state_length = 1) +
  enter_fade() +
  exit_fade() +
  labs(subtitle = "Day {closest_state}")

# Render the animation
animate(
  anim,
  width = 800, 
  height = 600, 
  renderer = gifski_renderer("maps/movementBlue.gif")
)