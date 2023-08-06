from utils.recordings.plots import plots_for_path, plot_success_rate

if __name__ == '__main__':  
    plots_for_path(folder_path="recordings/freq_tests/path_v0",
                   dist_filename="dist_5.png",
                   time_filename="time_5.png",
                   constant_key="uav_velocity",
                   constant_value=5,
                   mode="binary",
                   path_version="v0",
                   nn_name="SSD"
    )
    plot_success_rate(folder_path="recordings/freq_tests/path_v0",
                      out_filename="recordings/freq_tests/path_v0/success_rate.png",
                      path_version="v0",
                      constant_key="uav_velocity",
                      constant_value=5
    )
