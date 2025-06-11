import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr

# Load the CSV data
csv_path = "evaluation_results.csv"
df = pd.read_csv(csv_path)

print(df)

# Convert scores to float just in case
df["Self-BLEU"] = df["Self-BLEU"].astype(float)
df["Relevance Score"] = df["Relevance Score"].astype(float)
df["Groundedness Score"] = df["Groundedness Score"].astype(float)

# Compute summary statistics
summary_stats = df[["Self-BLEU", "Relevance Score", "Groundedness Score"]].describe()

# Calculate Pearson correlations
correlation_matrix = df[["Self-BLEU", "Relevance Score", "Groundedness Score"]].corr(method="pearson")

# Calculate how many are verifiable
verifiability_count = df["Verifiable?"].value_counts()

# # Visualization: Distributions and scatter plots
# plt.figure(figsize=(12, 6))
# sns.histplot(df["Self-BLEU"], kde=True).set_title("Distribution of Self-BLEU Scores")
# plt.show()

# plt.figure(figsize=(12, 6))
# sns.histplot(df["Relevance Score"], kde=True).set_title("Distribution of Relevance Scores")
# plt.show()

# plt.figure(figsize=(12, 6))
# sns.histplot(df["Groundedness Score"], kde=True).set_title("Distribution of Groundedness Scores")
# plt.show()

# Scatter plot to show relationship between relevance and groundedness
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="Relevance Score", y="Groundedness Score")
plt.title("Relevance vs Groundedness")
plt.show()

# Basic descriptive statistics
summary_stats = df[["Self-BLEU", "Relevance Score", "Groundedness Score"]].describe()

# Correlation matrix
correlation_matrix = df[["Self-BLEU", "Relevance Score", "Groundedness Score"]].corr()

# Count of verifiability
verifiability_counts = df["Verifiable?"].value_counts()

# # Plotting distributions
# plt.figure(figsize=(12, 4))
# for i, metric in enumerate(["Self-BLEU", "Relevance Score", "Groundedness Score"]):
#     plt.subplot(1, 3, i+1)
#     sns.histplot(df[metric], bins=10, kde=True)
#     plt.title(f"Distribution of {metric}")
# plt.tight_layout()
# plt.show()

# Set a style for better aesthetics
sns.set_style("whitegrid")

# Create a figure and a set of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6)) # Increased figure size for better readability

# Define metrics and their more descriptive titles
metrics_to_plot = {
    "Self-BLEU": "Diversity of Responses (Self-BLEU Score)",
    "Relevance Score": "Pertinence to Prompt (Relevance Score)",
    "Groundedness Score": "Factual Support (Groundedness Score)"
}

# Loop through each metric to create a detailed histogram
for i, (metric_col, plot_title) in enumerate(metrics_to_plot.items()):
    ax = axes[i] # Get the current subplot axis

    # Create the histogram with enhanced features
    sns.histplot(
        df[metric_col],
        bins=10,             # Number of bins for the histogram
        kde=True,            # Add Kernel Density Estimate (smooth curve)
        color=sns.color_palette("viridis")[i], # Use different colors for each plot
        edgecolor='black',   # Add a black edge to the bars for clarity
        linewidth=0.8,       # Line width for bar edges
        alpha=0.7,           # Transparency of the bars
        ax=ax                # Specify the subplot axis
    )

    # Add a vertical line for the mean
    mean_val = df[metric_col].mean()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')

    # Add descriptive labels and titles
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.set_xlabel(metric_col, fontsize=12)
    ax.set_ylabel('Number of Responses', fontsize=12)
    ax.legend() # Show the legend for the mean line

    # Add a grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout to prevent overlapping titles and labels
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for a potential super title

# Add a overall title for the figure
fig.suptitle('LLM Performance Metrics: Distribution Analysis', fontsize=18, y=1.02) # y adjusts vertical position

# Save the plot to a file
# plt.savefig('llm_performance_distributions_enhanced.png', dpi=300, bbox_inches='tight')
# plt.show()


summary_stats, correlation_matrix, verifiability_count
