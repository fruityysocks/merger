install.packages("maps")
library(maps)

# Now plot all the data points from a file called
# myfile, which has childLat, childLong coords
# as decimal GIS coords
# Center this map in New England
# and include state boundary lines
x <- read.csv("data/demographicsJittered.csv")
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
jpeg(filename = sprintf("maps/%s.jpg", timestamp), res=1200, width=6000, height=6000)
 
map("state", xlim=c(-73.5, -65), ylim=c(41.15, 48), fill=FALSE)
points(x$childLongJittered, x$childLatJittered, pch=16, cex=0.5)
title("Day 1")
dev.off()