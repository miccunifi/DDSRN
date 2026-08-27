import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------
# DATA LOADING & EXTRACTION
# ---------------------------------------------------------

def load_corruption_data(json_file):
    """Load corruption evaluation results from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def extract_metrics_for_plots(data):
    """
    Extracts metrics for Severity 1-5 AND adds Severity 0 (Clean).
    Returns a Pandas DataFrame.
    """
    all_data = []
    
    # --- 1. Extract Corruption Data (Severity 1-5) ---
    if 'corruptions' in data:
        for corruption, severity_data in data['corruptions'].items():
            for severity_str, metrics in severity_data.items():
                try:
                    severity = int(severity_str)
                except ValueError:
                    continue 
                
                if not metrics or 'mAP' not in metrics:
                    continue
                    
                entry = {
                    'severity': severity,
                    'corruption': corruption,
                    'mAP': metrics['mAP']
                }
                
                # Handle metric name variations
                if 'mean_dds' in metrics:
                    entry['dds'] = metrics['mean_dds']
                elif 'dds' in metrics:
                    entry['dds'] = metrics['dds']
                else:
                    entry['dds'] = np.nan

                if 'mean_lpips' in metrics:
                    entry['lpips'] = metrics['mean_lpips']
                elif 'lpips' in metrics:
                    entry['lpips'] = metrics['lpips']
                else:
                    entry['lpips'] = np.nan
                
                all_data.append(entry)
    
    # --- 2. Extract Clean Data (Severity 0) ---
    if 'clean' in data and 'mAP' in data['clean']:
        all_data.append({
            'severity': 0,
            'corruption': 'Clean Baseline',
            'mAP': data['clean']['mAP'],
            'dds': 0.0,      # Baseline distance is 0
            'lpips': 0.0     # Baseline distance is 0
        })
    
    return pd.DataFrame(all_data)

# ---------------------------------------------------------
# CALCULATION FUNCTIONS
# ---------------------------------------------------------

def compute_correlations_by_severity(df):
    """Compute GLOBAL Pearson correlations for each severity level (aggregating all corruptions)"""
    correlations = {}
    valid_severities = sorted(df[df['severity'] > 0]['severity'].unique())
    
    for severity in valid_severities:
        severity_df = df[df['severity'] == severity].dropna()
        
        if len(severity_df) > 1:
            dds_corr = np.nan
            if 'dds' in severity_df.columns and severity_df['dds'].std() > 0:
                dds_corr, _ = pearsonr(severity_df['dds'], severity_df['mAP'])
            
            lpips_corr = np.nan
            if 'lpips' in severity_df.columns and severity_df['lpips'].std() > 0:
                lpips_corr, _ = pearsonr(severity_df['lpips'], severity_df['mAP'])
            
            correlations[severity] = {
                'dds_map_correlation': dds_corr,
                'lpips_map_correlation': lpips_corr,
                'n_samples': len(severity_df)
            }
    return correlations

def compute_per_distortion_correlations(df):
    """
    Computes Pearson correlation (mAP vs Metric) for EACH distortion type separately,
    using the trend across Severities 1-5.
    """
    results = []
    unique_corruptions = df[df['severity'] > 0]['corruption'].unique()
    
    for corr in unique_corruptions:
        # We strictly use Severities > 0 to measure the degradation trend
        sub_df = df[(df['corruption'] == corr) & (df['severity'] > 0)].dropna()
        
        # Need at least 2 points (and some variance) to calculate correlation
        if len(sub_df) < 2:
            continue
            
        # 1. DDS Correlation
        r_dds = np.nan
        if 'dds' in sub_df.columns and sub_df['dds'].std() > 0:
             r_dds, _ = pearsonr(sub_df['mAP'], sub_df['dds'])
        
        # 2. LPIPS Correlation
        r_lpips = np.nan
        if 'lpips' in sub_df.columns and sub_df['lpips'].std() > 0:
             r_lpips, _ = pearsonr(sub_df['mAP'], sub_df['lpips'])
             
        results.append({'Corruption': corr, 'Metric': 'DDS', 'Correlation': r_dds})
        results.append({'Corruption': corr, 'Metric': 'LPIPS', 'Correlation': r_lpips})
        
    return pd.DataFrame(results)

# ---------------------------------------------------------
# PLOTTING FUNCTIONS
# ---------------------------------------------------------

def create_mean_map_plot(df, output_dir):
    """Creates a plot showing ONLY the Mean mAP per severity (Aggregated)."""
    mean_df = df.groupby('severity')['mAP'].mean().reset_index()
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=mean_df, x='severity', y='mAP', marker='o', markersize=10, 
                 linewidth=3, color='#2ca02c')
    
    for index, row in mean_df.iterrows():
        plt.text(row['severity'], row['mAP'] + 0.01, f"{row['mAP']:.3f}", 
                 ha='center', fontsize=11, fontweight='bold')

    plt.title('Mean mAP vs Severity (All Corruptions Averaged)', fontsize=14, fontweight='bold')
    plt.xlabel('Severity Level', fontsize=12, fontweight='bold')
    plt.ylabel('Mean MAP', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(mean_df['severity'].unique())
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_map_plot.png', dpi=300)
    plt.close()

def create_per_corruption_map_plot(df, output_dir):
    """Creates a single plot with multiple lines (one per corruption)."""
    clean_row = df[df['severity'] == 0]
    if clean_row.empty: return
    clean_map = clean_row.iloc[0]['mAP']
    
    corruption_df = df[df['severity'] > 0].copy()
    if corruption_df.empty: return

    # Normalize start points
    unique_corruptions = corruption_df['corruption'].unique()
    start_points = []
    for corr in unique_corruptions:
        start_points.append({'severity': 0, 'corruption': corr, 'mAP': clean_map})
    
    plot_df = pd.concat([pd.DataFrame(start_points), corruption_df], ignore_index=True)
    
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=plot_df, x='severity', y='mAP', hue='corruption', 
                 style='corruption', markers=True, dashes=False, linewidth=2, palette='tab20')
    
    plt.title('mAP Degradation per Distortion Type', fontsize=16, fontweight='bold')
    plt.xlabel('Severity Level', fontsize=14, fontweight='bold')
    plt.ylabel('mAP', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xticks([0, 1, 2, 3, 4, 5])
    plt.legend(title='Corruption Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'per_corruption_map_plot.png', dpi=300)
    plt.close()

def create_correlation_plots(correlations, output_dir):
    """
    Plot GLOBAL correlation coefficients vs Severity.

    Label rule:
    - If DDSRN is higher than LPIPS at a given severity, DDSRN label goes above
      and LPIPS label goes below.
    - If LPIPS is higher, LPIPS label goes above and DDSRN label goes below.
    - If the two values are very close, labels are also shifted left/right.
    """
    severities = sorted(correlations.keys())

    valid_severities = []
    for s in severities:
        dds_val = correlations[s].get('dds_map_correlation', np.nan)
        lpips_val = correlations[s].get('lpips_map_correlation', np.nan)

        if not np.isnan(dds_val) or not np.isnan(lpips_val):
            valid_severities.append(s)

    if not valid_severities:
        return

    dds_corrs = [
        correlations[s].get('dds_map_correlation', np.nan)
        for s in valid_severities
    ]

    lpips_corrs = [
        correlations[s].get('lpips_map_correlation', np.nan)
        for s in valid_severities
    ]

    plt.figure(figsize=(9, 5))

    plt.plot(
        valid_severities,
        dds_corrs,
        'o-',
        linewidth=3,
        markersize=10,
        color='#2E86AB',
        label='DDSRN Correlation',
        markerfacecolor='#2E86AB',
        markeredgecolor='white',
        markeredgewidth=2,
        zorder=3,
    )

    plt.plot(
        valid_severities,
        lpips_corrs,
        's-',
        linewidth=3,
        markersize=10,
        color='#A23B72',
        label='LPIPS Correlation',
        markerfacecolor='#A23B72',
        markeredgecolor='white',
        markeredgewidth=2,
        zorder=3,
    )

    plt.xlabel('Severity', fontsize=18, fontweight='bold')
    plt.ylabel('Pearson Correlation', fontsize=18, fontweight='bold')
    plt.title(
        'Correlation with mAP across Severities',
        fontsize=18,
        fontweight='bold',
        pad=16,
    )

    plt.xticks(valid_severities)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.grid(True, alpha=0.3, linestyle='--', zorder=0)
    plt.axhline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)

    all_corrs = [
        v for v in dds_corrs + lpips_corrs
        if not np.isnan(v)
    ]

    if all_corrs:
        ymin = min(all_corrs)
        ymax = max(all_corrs)
        pad = max(0.16, 0.18 * (ymax - ymin))
        plt.ylim(max(-1.20, ymin - pad), min(1.10, ymax + pad))

    close_threshold = 0.10

    def annotate_score(x, y, text, color, vertical_side, horizontal_side=0):
        """
        vertical_side:
            +1 means label above point
            -1 means label below point

        horizontal_side:
            -1 means shift left
            0 means no shift
            +1 means shift right
        """
        x_offset = 16 * horizontal_side
        y_offset = 18 if vertical_side > 0 else -22
        va = 'bottom' if vertical_side > 0 else 'top'

        plt.annotate(
            text,
            (x, y),
            xytext=(x_offset, y_offset),
            textcoords='offset points',
            ha='center',
            va=va,
            fontsize=11,
            color=color,
            fontweight='bold',
            zorder=5,
        )

    for s, dds_corr, lpips_corr in zip(valid_severities, dds_corrs, lpips_corrs):
        dds_valid = not np.isnan(dds_corr)
        lpips_valid = not np.isnan(lpips_corr)

        if dds_valid and lpips_valid:
            are_close = abs(dds_corr - lpips_corr) < close_threshold

            # Higher score gets label above; lower score gets label below.
            if dds_corr >= lpips_corr:
                dds_vertical = +1
                lpips_vertical = -1
            else:
                dds_vertical = -1
                lpips_vertical = +1

            # If close, also move them to opposite horizontal sides.
            # Keep DDSRN on the left and LPIPS on the right for consistency.
            dds_horizontal = 0
            lpips_horizontal = 0

            annotate_score(
                s,
                dds_corr,
                f'{dds_corr:.2f}',
                '#2E86AB',
                vertical_side=dds_vertical,
                horizontal_side=dds_horizontal,
            )

            annotate_score(
                s,
                lpips_corr,
                f'{lpips_corr:.2f}',
                '#A23B72',
                vertical_side=lpips_vertical,
                horizontal_side=lpips_horizontal,
            )

        else:
            # Single valid value: place label away from center.
            if dds_valid:
                annotate_score(
                    s,
                    dds_corr,
                    f'{dds_corr:.2f}',
                    '#2E86AB',
                    vertical_side=+1,
                    horizontal_side=0,
                )

            if lpips_valid:
                annotate_score(
                    s,
                    lpips_corr,
                    f'{lpips_corr:.2f}',
                    '#A23B72',
                    vertical_side=-1,
                    horizontal_side=0,
                )

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=12,
        frameon=True,
    )

    plt.tight_layout()

    plt.savefig(
        output_dir / 'global_correlation_plots.png',
        dpi=300,
        bbox_inches='tight',
    )

    plt.close()

def create_box_plots(df, output_dir):
    """Create distributions (Excluding Sev 0)."""
    plot_df = df[df['severity'] > 0]
    if plot_df.empty: return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    metrics = [('dds', 'Blues', 'DDSRN'), ('lpips', 'Reds', 'LPIPS'), ('mAP', 'Greens', 'MAP')]
    #metrics = [('dds', 'Blues', 'DDS'), ('mAP', 'Greens', 'MAP')]
    
    for ax, (col, pal, title) in zip(axes, metrics):
        if col in plot_df.columns:
            sub_df = plot_df.dropna(subset=[col])
            if not sub_df.empty:
                sns.boxplot(data=sub_df, x='severity', y=col, ax=ax, palette=pal, width=0.6)
                sns.stripplot(data=sub_df, x='severity', y=col, ax=ax, color='black', alpha=0.5, jitter=True)
                ax.set_title(f'{title} Distribution', fontsize=20, fontweight='bold')
                ax.set_xlabel('Severity', fontsize=20)
                ax.set_ylabel(title, fontsize=20, fontweight='bold')
                ax.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'box_plots.png', dpi=300)
    plt.close()

def create_per_corruption_relationship_plots(df, output_dir):
    """
    Creates a Dual-Axis plot for EVERY corruption type showing the trend lines.
    """
    unique_corruptions = df[df['severity'] > 0]['corruption'].unique()
    
    clean_row = df[df['severity'] == 0]
    if clean_row.empty:
        clean_map = 0; clean_dds = 0; clean_lpips = 0
    else:
        clean_map = clean_row.iloc[0]['mAP']
        clean_dds = 0.0; clean_lpips = 0.0

    rel_dir = output_dir / "relationships"
    rel_dir.mkdir(exist_ok=True)

    for corr in unique_corruptions:
        sub_df = df[df['corruption'] == corr].sort_values('severity')
        
        if 0 not in sub_df['severity'].values:
            clean_entry = pd.DataFrame([{'severity': 0, 'corruption': corr, 'mAP': clean_map, 'dds': clean_dds, 'lpips': clean_lpips}])
            sub_df = pd.concat([clean_entry, sub_df]).sort_values('severity')
            
        stats_df = sub_df[sub_df['severity'] > 0]
        corr_title_parts = []
        if len(stats_df) > 2:
            if 'dds' in stats_df.columns and stats_df['dds'].std() > 0:
                r_dds, _ = pearsonr(stats_df['mAP'], stats_df['dds'])
                corr_title_parts.append(f"DDS R={r_dds:.2f}")
            if 'lpips' in stats_df.columns and stats_df['lpips'].std() > 0:
                r_lpips, _ = pearsonr(stats_df['mAP'], stats_df['lpips'])
                corr_title_parts.append(f"LPIPS R={r_lpips:.2f}")
        
        stats_text = " | ".join(corr_title_parts)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color_map = '#2ca02c' 
        ax1.set_xlabel('Severity Level', fontsize=12, fontweight='bold')
        ax1.set_ylabel('mAP (Detection)', color=color_map, fontsize=12, fontweight='bold')
        l1 = ax1.plot(sub_df['severity'], sub_df['mAP'], marker='o', color=color_map, 
                 linewidth=3, label='mAP')
        ax1.tick_params(axis='y', labelcolor=color_map)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_xticks([0, 1, 2, 3, 4, 5])
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Perception Distance (Lower is Better)', color='#1f77b4', fontsize=12, fontweight='bold')
        
        lines = l1
        if 'dds' in sub_df.columns:
            l2 = ax2.plot(sub_df['severity'], sub_df['dds'], marker='s', color='#1f77b4', 
                     linestyle='-', linewidth=2, label='DDS')
            lines += l2
        if 'lpips' in sub_df.columns:
            l3 = ax2.plot(sub_df['severity'], sub_df['lpips'], marker='^', color='#d62728', 
                     linestyle='--', linewidth=2, label='LPIPS')
            lines += l3
            
        ax2.tick_params(axis='y', labelcolor='black')
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
        title_str = f"{corr.replace('_', ' ').title()}\n{stats_text}"
        plt.title(title_str, y=1.15, fontsize=14, fontweight='bold')
        plt.tight_layout()
        safe_name = corr.replace(" ", "_")
        plt.savefig(rel_dir / f'relationship_{safe_name}.png', dpi=150)
        plt.close()

def plot_correlation_bar_chart(corr_df, output_dir):
    """
    Plots a Grouped Bar Chart showing the correlation coefficient for every distortion.
    """
    if corr_df.empty:
        print("No correlation data available for bar chart.")
        return

    plt.figure(figsize=(14, 8))
    
    # Sort by DDS correlation for visual cleanliness
    try:
        pivot_df = corr_df.pivot(index='Corruption', columns='Metric', values='Correlation')
        if 'DDS' in pivot_df.columns:
            pivot_df = pivot_df.sort_values('DDS', ascending=True) 
        sorted_corruptions = pivot_df.index.tolist()
    except Exception:
        sorted_corruptions = None
    
    sns.barplot(data=corr_df, x='Corruption', y='Correlation', hue='Metric', 
                order=sorted_corruptions, palette={'DDS': '#1f77b4', 'LPIPS': '#d62728'},
                edgecolor='black')
    
    plt.axhline(0, color='black', linewidth=1.5)
    plt.title('Correlation (mAP vs Perception) per Distortion Type', fontsize=16, fontweight='bold')
    plt.xlabel('Distortion Type', fontsize=14, fontweight='bold')
    plt.ylabel('Pearson Correlation Coefficient (r)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # --- RESTRICT Y-AXIS TO 0 to -1 ---
    # We use -1.05 to 0.05 to allow small margin for visibility
    plt.ylim(-1.05, 0.05) 
    
    plt.legend(title='Metric', loc='lower right')
    
    plt.tight_layout()
    save_path = output_dir / 'correlation_bar_chart.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Bar chart saved to {save_path}")

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    # --- CONFIGURATION ---
    input_json = "COCO_results_DDSRN.json"
    output_dir = Path("plots/COCO_plots/ECCV_rebuttal")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading {input_json}...")
    try:
        data = load_corruption_data(input_json)
    except FileNotFoundError:
        print(f"Error: File {input_json} not found!")
        return

    print("Extracting metrics...")
    df = extract_metrics_for_plots(data)
    
    if df.empty:
        print("No valid data found to plot.")
        return

    print(f"Found {len(df)} data points.")
    
    # 1. Global Stats
    corrs = compute_correlations_by_severity(df)
    
    # 2. Per-Distortion Stats
    corr_bar_df = compute_per_distortion_correlations(df)
    
    print("Generating Global plots...")
    create_correlation_plots(corrs, output_dir)
    create_box_plots(df, output_dir)
    create_mean_map_plot(df, output_dir)
    create_per_corruption_map_plot(df, output_dir)
    
    print("Generating per-corruption relationship plots...")
    create_per_corruption_relationship_plots(df, output_dir)
    
    print("Generating Correlation Bar Chart...")
    plot_correlation_bar_chart(corr_bar_df, output_dir)
    
    print(f"Done! Plots saved to {output_dir}/")

if __name__ == '__main__':
    main()