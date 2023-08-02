from utils.simulation import compare_runs, get_noisy_measurements, plot_clean_and_noisy_signals

if __name__ == '__main__':
    root_paths = ["recordings/20230801_195333",
                  "recordings/20230801_200731"]
    compare_runs(root_paths)

    target_snr = 100
    meas_key = "sim_lead_pos"
    sig, noisy_sig, times, achieved_snr = get_noisy_measurements(root_paths[0], target_snr, meas_key)
    print(f"Target SNR: {target_snr}")
    print(f"SNR achieved: {achieved_snr}")

    plot_clean_and_noisy_signals(sig, noisy_sig, times, "temp.png", "LeadingUAV position (s)")
