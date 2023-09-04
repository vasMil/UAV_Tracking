from pruning import load_pruning_report, plot_pruning_pareto

if __name__ == '__main__':
    stats = load_pruning_report("nets/checkpoints/pruning/ssd_pretrained/finetuning/report.json")
    plot_pruning_pareto(stats, "nets/checkpoints/pruning/ssd_pretrained/finetuning/pareto_front.svg")
