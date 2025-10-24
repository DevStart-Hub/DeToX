library(hdf5r)

# Read the file
file <- H5File$new("pytables_only.h5", mode = "r")

# Read gaze data table
gaze_data <- file[["gaze_data"]]$read()

# Convert to data frame
gaze_df <- as.data.frame(gaze_data)

# Print the results
print("Gaze data:")
print(head(gaze_df))

print(paste("Number of rows:", nrow(gaze_df)))
print(paste("Column names:", paste(names(gaze_df), collapse = ", ")))

# Close the file
file$close()