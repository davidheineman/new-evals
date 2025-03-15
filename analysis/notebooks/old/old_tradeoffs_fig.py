from metaanalysis import get_title_from_task
from plot import TASK_CATEGORIES, CATEGORY_COLORS, adjustText, get_valid_points, draw_pareto_frontier

def create_scatter_subplot(ax: plt.Axes, df, x_col, y_col, xlabel, ylabel, title, percentage=False, invert_x=False, invert_y=False, log_x=False, xlim=None, ylim=None):
    points = get_valid_points(df, x_col, y_col)
    if not points:
        return
    
    xs, ys, tasks = zip(*points)
    
    # Filter out -inf values if needed
    if log_x:
        valid_indices = [i for i in range(len(xs)) if xs[i] != float('-inf') and ys[i] != float('-inf')]
        xs = [xs[i] for i in valid_indices]
        ys = [ys[i] for i in valid_indices]
        tasks = [tasks[i] for i in valid_indices]
    
    colors = [CATEGORY_COLORS[TASK_CATEGORIES.get(task, 'knowledge')] for task in tasks]
    ax.scatter(xs, ys, s=4, c=colors)

    # Draw separate Pareto frontiers for each task category
    for category in set(TASK_CATEGORIES.values()):
        # Get points for this category
        category_points = [(x, y) for x, y, task in zip(xs, ys, tasks) if TASK_CATEGORIES.get(task, 'knowledge') == category]
        if category_points:
            cat_xs, cat_ys = zip(*category_points)
            draw_pareto_frontier(ax, cat_xs, cat_ys, invert_x=invert_x, invert_y=invert_y, color=CATEGORY_COLORS[category])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if log_x:
        ax.set_xscale('log')
    if invert_x:
        ax.invert_xaxis()
    if invert_y:
        ax.invert_yaxis()
    if xlim is not None:
        ax.set_xlim(**xlim)
    if ylim is not None:
        ax.set_ylim(**ylim)

    if percentage:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        if not log_x:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
    
    texts = []
    for x, y, task in zip(xs, ys, tasks):
        # Only add text if point is within axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if ((xlim[0] <= x <= xlim[1]) if not ax.xaxis_inverted() else (xlim[1] <= x <= xlim[0])) and \
           ((ylim[0] <= y <= ylim[1]) if not ax.yaxis_inverted() else (ylim[1] <= y <= ylim[0])):
            texts += [ax.text(x, y, get_title_from_task(task), fontsize=5, clip_on=True)]
        
    adjustText(ax, texts)
    return ax

def create_bar_subplot(ax: plt.Axes, df, col, ylabel, title, top_perc=1, percentage=False, sort_ascending=True):
    valid_tasks = df[df[col].notna()].index
    y_vals = df.loc[valid_tasks, col]
    task_names = [get_title_from_task(t) for t in valid_tasks]
    
    sorted_indices = np.argsort(y_vals)
    if not sort_ascending:
        sorted_indices = sorted_indices[::-1]

    valid_tasks = valid_tasks[sorted_indices]
    y_vals = y_vals[sorted_indices]
    task_names = [task_names[i] for i in sorted_indices]

    # Only keep the top_perc% of benchmarks
    import math
    valid_tasks = valid_tasks[math.floor(len(valid_tasks)*(1-top_perc)):]
    y_vals = y_vals[math.floor(len(y_vals)*(1-top_perc)):]
    task_names = task_names[math.floor(len(task_names)*(1-top_perc)):]
        
    colors = [CATEGORY_COLORS[TASK_CATEGORIES.get(task, 'knowledge')] for task in valid_tasks]
    ax.bar(range(len(valid_tasks)), y_vals, color=colors)
    ax.set_xticks(range(len(valid_tasks)))
    ax.set_xticklabels(task_names, rotation=45, ha='right', fontsize=6)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if percentage:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    return ax

# fig, axs = plt.subplots(2, 3, figsize=(16, 10))
# axs = axs.flatten()

mosaic = """AABBCC
            .DDEE."""
# fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(26, 11))
fig, ax_dict = plt.subplot_mosaic(mosaic, figsize=(26, 9))
axs = list(ax_dict.values())
ax_dict['B'].tick_params('y', labelleft=True)
ax_dict['C'].tick_params('y', labelleft=True)
ax_dict['D'].tick_params('y', labelleft=True)
ax_dict['E'].tick_params('y', labelleft=True)

# Subplot 1
create_scatter_subplot(
    axs[0], df_results,
    x_col='two_class_bpb_60M', 
    y_col='mean_error_step_2_external',
    xlabel='Decision Acc. BPB (60M)', 
    ylabel='Task Loss Mean Error on External Models',
    # title='> 1000s models (~0.001% target compute)',
    title='Data Mixing Experiments',
    percentage=True,
    invert_y=True,
    # xlim={'left': 0.6, 'right': 1},
    # ylim={'top': 0, 'bottom': 0.05}

    # ylim={'top': 0, 'bottom': 0.08}
)

# Subplot 2
create_scatter_subplot(
    axs[1], df_results,
    x_col='two_class_bpb_60M', 
    y_col='perc_sig_1B_bpb',
    xlabel='Decision Acc. BPB (60M)', 
    ylabel='% Sig Comparisons on BPB (DataDecide 1B)',
    # title='~ 100s models @ Pre-train (~0.1% target compute)',
    title='Pre-training Data Experiments',
    percentage=True,
    # xlim={'left': 0.6, 'right': 1},
    # ylim={'bottom': 0.8, 'top': 1}

    xlim={'right': 1},
    ylim={'top': 1}
)

# Subplot 3
create_scatter_subplot(
    axs[2], df_results,
    x_col='rel_error_step_1_7b',
    y_col='rel_error_stacked_7b',
    xlabel='Task Loss Relative Error (predicting 7B-4T)',
    ylabel='Stacked Relative Error (predicting 7B-4T)',
    # title='< 10 models (~1% target compute)',
    title='Scaling Law Prediction',
    percentage=True,
    invert_x=True,
    invert_y=True,
    # xlim={'left': 0.05, 'right': 0},
    # ylim={'bottom': 0.05, 'top': 0}

    ylim={'bottom': 0.2, 'top': 0}
)
# axs[2].set_ylim(0, 0.10)

# Subplot 3
create_scatter_subplot(
    axs[3], df_results,
    x_col='total_cost',
    y_col='perc_sig_1B_bpb',
    xlabel='Total Cost (# forward passes)',
    ylabel='% Sig Comparisons on BPB (DataDecide 1B)',
    # title='1 model (in-loop)',
    title='Monitoring a Training Run',
    percentage=True,
    invert_x=True,
    log_x=True,
    # ylim={'bottom': 0.8, 'top': 1}
    ylim={'bottom': 0.2, 'top': 1}
)
axs[3].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

# Subplot 4
create_scatter_subplot(
    axs[4], df_results,
    x_col='perc_sig_1B_bpb',
    y_col='perc_sig_1B_primary_score',
    xlabel='% Sig Comparisons on BPB (DataDecide 1B)',
    ylabel='% Sig Comparisons on Primary Score (DataDecide 1B)',
    # title='~ 100s models @ Mid-train (~4% target compute)',
    title='Mid-training Data Experiments',
    # top_perc=0.3,
    percentage=True,
    # xlim={'left': 0.8, 'right': 1},
    # ylim={'bottom': 0.2, 'top': 1}
    xlim={'left': 0, 'right': 1},
    ylim={'bottom': 0, 'top': 1}
)
# axs[4].set_ylim(bottom=0.8, top=1)

# # Subplot 6
# create_scatter_subplot(
#     axs[5], df_results,
#     x_col='total_cost',
#     y_col='rel_error_stacked_7b',
#     xlabel='Total Cost (# forward passes)',
#     ylabel='Stacked Relative Error (predicting 7B-4T)',
#     title='1 model (in-loop, predictive)',
#     percentage=True,
#     invert_y=True,
#     invert_x=True,
#     log_x=True,
#     ylim={'top': 0, 'bottom': 0.05}
# )
# axs[5].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))

# Create legend patches
legend_patches = [plt.Rectangle((0,0),1,1, fc=color) for color in CATEGORY_COLORS.values()]
legend_patches.append(plt.Line2D([0], [0], linestyle=':', color='grey'))
legend_labels = [key.capitalize() for key in CATEGORY_COLORS.keys()]
legend_labels.append('Decision-Optional Frontier')

# Add single legend at bottom of figure
fig.legend(legend_patches, legend_labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=len(legend_labels), fontsize=12) # -0.02

plt.subplots_adjust(left=0.2, right=0.8, wspace=0.7, hspace=0.35) # hspace=0.5
# plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/tradeoffs.pdf', bbox_inches='tight', dpi=3000)
plt.show()