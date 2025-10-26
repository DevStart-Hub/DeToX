library(hdf5r)

# Read the file
file <- H5File$new("TEST4.h5", mode = "r")

# Read gaze and event data table
gaze_data <- file[["gaze"]]$read()
event_data <- file[["events"]]$read()


# Close the file
file$close()