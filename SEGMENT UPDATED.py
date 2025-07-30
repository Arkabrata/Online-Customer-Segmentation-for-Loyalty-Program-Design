# %%
import pandas as pd
df=pd.read_csv('/Users/rarerabbit/Documents/Data Analysis/segment/online customer_data_2.csv')
print(df.head())

# %%
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

# %%
#Renaming columns for better readability
df.rename(columns={
    'PCT_RTO_ORDS': 'RTO%',
    'PCT_CIR_ORDS': 'RTV%',
    'grossordercount': 'Gross Sold Qty',
    'frequencyoforder': 'Order Frequency',
    'customerlifespanwithbrand': 'Lifespan',
    'averageordervalue': 'AOV',
    'onlineandofflinepresence': 'OMNI',
    'DISC': 'Discount%',
    'GP_value': 'Gross Profit',
    'APP_WEB': 'App/Web User'
    },inplace=True)
print(df.columns.tolist())

# %%
#statistical SUmmary
exclude_cols=['OMNI','CUSTOMER_MOBILE','App/Web User']

numeric_cols=[col for col in df.columns if col not in exclude_cols]
df1=df[numeric_cols]
summary_stats= df1.describe().T
summary_stats['Median']=df1.median(numeric_only=True)
summary_stats['Mode']=df1.mode(numeric_only=True).iloc[0]
summary_stats=summary_stats[['count','mean','Median','Mode','std','min','max']]
summary_stats.columns=['Count','Mean','Median','Mode','Std Dev','Min','Max']
summary_stats=summary_stats.round(2)
print(summary_stats.to_string())


# %%
df['Discount%'] = df['Discount%'].clip(lower=0, upper=100)


# %%
#statistical SUmmary
exclude_cols=['OMNI','CUSTOMER_MOBILE','App/Web User']

numeric_cols=[col for col in df.columns if col not in exclude_cols]
df1=df[numeric_cols]
summary_stats= df1.describe().T
summary_stats['Median']=df1.median(numeric_only=True)
summary_stats['Mode']=df1.mode(numeric_only=True).iloc[0]
summary_stats=summary_stats[['count','mean','Median','Mode','std','min','max']]
summary_stats.columns=['Count','Mean','Median','Mode','Std Dev','Min','Max']
summary_stats=summary_stats.round(2)
print(summary_stats.to_string())



# %%
from scipy.stats import skew, kurtosis

# Select only numeric features (excluding CUSTOMER_MOBILE and OMNI)
exclude = ['CUSTOMER_MOBILE', 'OMNI', 'App/Web User']
numeric_cols = [col for col in df.columns if col not in exclude]

# Compute skewness and kurtosis
skew_vals = df[numeric_cols].apply(skew)
kurt_vals = df[numeric_cols].apply(kurtosis)

# Combine into a summary table
sk_summary = pd.DataFrame({
    'Skewness': skew_vals.round(2),
    'Kurtosis': kurt_vals.round(2)
})

print("Skewness & Kurtosis Summary:\n")
print(sk_summary)


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis

# Columns to exclude from analysis
exclude_cols = ['CUSTOMER_MOBILE', 'OMNI', 'App/Web User']
numeric_cols = [col for col in df1.columns if col not in exclude_cols]
df_numeric = df1[numeric_cols]

# Skewness: Distribution plots
sns.set(style="whitegrid")
plt.figure(figsize=(15, 3 * len(df_numeric.columns)))

for i, col in enumerate(df_numeric.columns, 1):
    plt.subplot(len(df_numeric.columns), 1, i)
    sns.histplot(df_numeric[col], kde=True, color='steelblue', bins=50)
    plt.axvline(df_numeric[col].mean(), color='red', linestyle='--', label='Mean')
    skewness_val = skew(df_numeric[col].dropna())
    plt.title(f"Skewness of {col}: {skewness_val:.2f}")
    plt.xlabel(col)
    plt.legend()

plt.tight_layout()
skew_path = "/Users/rarerabbit/Documents/Data Analysis/segment/skewness_plot.jpg"
plt.savefig(skew_path, dpi=300)
print(f"✅ Skewness plot saved to: {skew_path}")
plt.close()

# Kurtosis: Bar chart
plt.figure(figsize=(10, 6))
kurtosis_values = df_numeric.apply(lambda x: kurtosis(x.dropna()))
sns.barplot(x=kurtosis_values.index, y=kurtosis_values.values, palette='Set2')
plt.title("Kurtosis of Features")
plt.ylabel("Kurtosis")
plt.xticks(rotation=45)
plt.tight_layout()

kurtosis_path = "/Users/rarerabbit/Documents/Data Analysis/segment/kurtosis_plot.jpg"
plt.savefig(kurtosis_path, dpi=300)
print(f"✅ Kurtosis plot saved to: {kurtosis_path}")
plt.close()


# %%
import numpy as np

# Apply log1p to positively skewed variables
df1['RTO%_log']        = np.log1p(df1['RTO%'])
df1['RTV%_log']        = np.log1p(df1['RTV%'])
df1['GrossQty_log']    = np.log1p(df1['Gross Sold Qty'])
df1['Freq_log']        = np.log1p(df1['Order Frequency'])
df1['Lifespan_log']    = np.log1p(df1['Lifespan'])

# AOV log with shift (if min <= 0)
aov_min = df1['AOV'].min()
aov_shift = abs(aov_min) + 1 if aov_min <= 0 else 0
df1['AOV_log'] = np.log1p(df1['AOV'] + aov_shift)

# Gross Profit log with shift (if min <= 0)
gp_min = df1['Gross Profit'].min()
gp_shift = abs(gp_min) + 1 if gp_min <= 0 else 0
df1['GrossProfit_log'] = np.log1p(df1['Gross Profit'] + gp_shift)


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Include all relevant log-transformed features
log_features = [
    'RTO%_log', 'RTV%_log', 'GrossQty_log',
    'Freq_log', 'Lifespan_log', 'AOV_log', 'GrossProfit_log'
]

# Generate pairplot
pairplot_fig = sns.pairplot(df1[log_features], plot_kws={'alpha': 0.6})
pairplot_fig.fig.set_size_inches(14, 10)
pairplot_fig.fig.suptitle("Pairplot: Log-Transformed Customer Features", y=1.03)

# Save and show
save_path = "/Users/rarerabbit/Documents/Data Analysis/segment/log_transformed_pairplot.jpg"
pairplot_fig.savefig(save_path, dpi=300)
plt.tight_layout()
plt.show()

print(f"Pairplot saved at: {save_path}")


# %%
import seaborn as sns
import matplotlib.pyplot as plt

# Include all log-transformed features
log_features = ['RTO%_log', 'RTV%_log', 'GrossQty_log', 'Freq_log', 'Lifespan_log', 'AOV_log', 'GrossProfit_log']

# Compute correlation matrix
corr_matrix = df1[log_features].corr()

# Plot the heatmap with better precision and annotation visibility
plt.figure(figsize=(10, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".3f",  # Increased precision to 3 decimal places
    square=True,
    annot_kws={"size": 10, "color": "black"}  # Black text for better readability
)
plt.title("Correlation Heatmap: Log-Transformed Features")
plt.tight_layout()

# Save the heatmap
heatmap_path = "/Users/rarerabbit/Documents/Data Analysis/segment/log_transformed_correlation_heatmap.jpg"
plt.savefig(heatmap_path, dpi=300)
plt.show()

print(f"Heatmap saved at: {heatmap_path}")


# %%
#Normalize the data
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Define the log-transformed features
log_features = ['RTO%_log', 'RTV%_log', 'GrossQty_log', 'Freq_log', 'Lifespan_log', 'AOV_log', 'GrossProfit_log']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform
df_scaled_log = pd.DataFrame(
    scaler.fit_transform(df1[log_features]),
    columns=[f"{col}_scaled" for col in log_features]
)

# Attach to the original dataframe
df1 = pd.concat([df1, df_scaled_log], axis=1)

# Check the result
print(df_scaled_log.head())


# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Define your scaled columns
scaled_cols = [
    'RTO%_log_scaled', 'RTV%_log_scaled', 'GrossQty_log_scaled',
    'Freq_log_scaled', 'Lifespan_log_scaled', 'AOV_log_scaled', 'GrossProfit_log_scaled'
]

# Plot histograms
plt.figure(figsize=(16, 20))
for i, col in enumerate(scaled_cols, 1):
    plt.subplot(len(scaled_cols), 1, i)
    sns.histplot(df1[col], bins=50, kde=True, color='skyblue')
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.savefig("/Users/rarerabbit/Documents/Data Analysis/segment/scaled_features_distribution.jpg", dpi=300)
plt.show()


# %%
plt.figure(figsize=(14, 6))
sns.boxplot(data=df1[scaled_cols], palette='Set3')
plt.title("Boxplot of Scaled Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/Users/rarerabbit/Documents/Data Analysis/segment/scaled_features_boxplot.jpg", dpi=300)
plt.show()

# %%
from sklearn.preprocessing import StandardScaler
import pandas as pd


df_log = df1[['RTO%_log', 'RTV%_log', 'GrossQty_log', 'Freq_log', 'Lifespan_log', 'AOV_log']]


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_log)


df_log_scaled = pd.DataFrame(scaled_data, columns=[
    'RTO%_log_scaled', 'RTV%_log_scaled', 'GrossQty_log_scaled',
    'Freq_log_scaled', 'Lifespan_log_scaled', 'AOV_log_scaled'
])


# %%
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Elbow Method
X = df1[scaled_cols]  # Make sure this contains your *_log_scaled columns
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.tight_layout()
elbow_path = "/Users/rarerabbit/Documents/Data Analysis/segment/elbow_curve.jpg"
plt.savefig(elbow_path, dpi=300)
plt.show()
print(f"Elbow plot saved at: {elbow_path}")


# %%
from sklearn.cluster import KMeans

# Step 2: Apply final KMeans model with optimal K=4
k_optimal = 4
kmeans_final = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df1['Customer_Cluster'] = kmeans_final.fit_predict(df1[scaled_cols])

# Optional: check distribution of customers across clusters
print(df1['Customer_Cluster'].value_counts())


# %%
cluster_profile = df1.groupby('Customer_Cluster')[[
    'RTO%', 'RTV%', 'Gross Sold Qty', 'Return_qty',
    'Order Frequency', 'Lifespan', 'AOV', 'Discount%', 'Gross Profit'
]].agg(['mean', 'median']).round(2)

print(cluster_profile)


# %%
from sklearn.cluster import KMeans

# Input: All scaled log-transformed features
features_for_clustering = df1[[
    'RTO%_log_scaled', 'RTV%_log_scaled', 'GrossQty_log_scaled',
    'Freq_log_scaled', 'Lifespan_log_scaled',
    'AOV_log_scaled', 'GrossProfit_log_scaled'
]]

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df1['Cluster'] = kmeans.fit_predict(features_for_clustering)


# %%
cluster_summary = df1.groupby('Cluster').agg({
    'RTO%': ['mean', 'median'],
    'RTV%': ['mean', 'median'],
    'Gross Sold Qty': ['mean', 'median'],
    'Return_qty': ['mean'],
    'Order Frequency': ['mean', 'median'],
    'Lifespan': ['mean', 'median'],
    'AOV': ['mean', 'median'],
    'Gross Profit': ['mean', 'median'],
    'Discount%': ['mean', 'median']
})


# %%
# Say cluster 2 is best, 1 is worst:
cluster_to_tier = {
    2: 'Platinum',
    0: 'Gold',
    3: 'Silver',
    1: 'Bronze'
}
df1['Customer_Tier_Label'] = df1['Cluster'].map(cluster_to_tier)


# %%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Use only scaled features
features = [
    'RTO%_log_scaled', 'RTV%_log_scaled', 'GrossQty_log_scaled',
    'Freq_log_scaled', 'Lifespan_log_scaled',
    'AOV_log_scaled', 'GrossProfit_log_scaled'
]

X = df1[features]

# Run PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X)

# Create a DataFrame for plotting
df_pca = pd.DataFrame(data=pca_components, columns=['PCA1', 'PCA2'])
df_pca['Cluster'] = df1['Cluster']

# Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Cluster', palette='tab10', alpha=0.7)
plt.title("PCA Projection of Customer Clusters")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("/Users/rarerabbit/Documents/Data Analysis/segment/pca_cluster_plot.jpg", dpi=300)
plt.show()


# %%
from sklearn.manifold import TSNE

# Run t-SNE on top 30 PCA components or directly on scaled features
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
tsne_results = tsne.fit_transform(X)

df_tsne = pd.DataFrame(tsne_results, columns=['Dim1', 'Dim2'])
df_tsne['Cluster'] = df1['Cluster']

# Plot t-SNE
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', hue='Cluster', palette='tab10', alpha=0.7)
plt.title("t-SNE Projection of Customer Clusters")
plt.tight_layout()
plt.savefig("/Users/rarerabbit/Documents/Data Analysis/segment/tsne_cluster_plot.jpg", dpi=300)
plt.show()


# %%
# Get the score boundaries for each tier
score_thresholds = df1.groupby('Customer_Tier_Label')['Customer_Score'].agg(['min', 'max']).reindex(['Bronze', 'Silver', 'Gold', 'Platinum'])
print("📊 Score Thresholds:\n", score_thresholds)


# %%
# Get average feature values per tier
feature_benchmarks = df1.groupby('Customer_Tier_Label')[['Order Frequency', 'AOV', 'Lifespan']].mean().round(2)
print("📌 Feature Benchmarks:\n", feature_benchmarks)


# %%
cluster_to_tier = {
    2: 'Platinum',
    0: 'Gold',
    1: 'Silver',
    3: 'Bronze'  # or 'Base'
}

df1['Loyalty_Tier'] = df1['Customer_Cluster'].map(cluster_to_tier)


# %%
import pandas as pd

# Step 1: Define Customer Score with Gross Profit having highest weight
df['Customer_Score'] = (
    0.4 * df['Gross Profit'] +           # Highest weight
    0.2 * df['AOV'] +
    0.15 * df['Order Frequency'] +
    0.1 * df['Lifespan'] -
    25 * df['RTO%'] -
    10 * df['RTV%'] -
    20 * df['Discount%']                 # Negative impact
)

# Step 2: Assign Tiers Based on Quantiles
df['Customer_Tier_Label'] = pd.qcut(
    df['Customer_Score'],
    q=[0, 0.70, 0.85, 0.95, 1.0],  # 70% Bronze, 15% Silver, 10% Gold, 5% Platinum
    labels=['Bronze', 'Silver', 'Gold', 'Platinum']
)

# Step 3: Customer count by tier
tier_counts = df['Customer_Tier_Label'].value_counts().sort_index()
print("Customer Counts by Tier:\n", tier_counts)



# %%
import matplotlib.pyplot as plt

tiers = ['Platinum', 'Gold', 'Silver', 'Bronze']
counts = [49285, 98568, 147853, 689980] 


tiers_reversed = tiers[::-1]
counts_reversed = counts[::-1]

# Plot horizontal bar chart
plt.figure(figsize=(10, 6))
bars = plt.barh(tiers_reversed, counts_reversed, color='skyblue')
plt.xlabel('Number of Customers')
plt.title('Customer Loyalty Tier Distribution (Inverted Funnel)')

# Annotate bars
for bar in bars:
    width = bar.get_width()
    plt.text(width + 2000, bar.get_y() + bar.get_height()/2,
             f'{int(width):,}', va='center', fontsize=10)

plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig("/mnt/data/loyalty_tier_funnel_chart.jpg", dpi=300)
plt.show()


# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Tier count for funnel chart
tier_counts = df['Customer_Tier_Label'].value_counts().reindex(['Bronze', 'Silver', 'Gold', 'Platinum'])

# Funnel Chart
plt.figure(figsize=(10, 6))
sns.barplot(x=tier_counts.values, y=tier_counts.index, palette='coolwarm')
plt.title("Loyalty Tier Funnel Chart")
plt.xlabel("Number of Customers")
plt.ylabel("Customer Tier")
plt.tight_layout()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.savefig("/Users/rarerabbit/Documents/Data Analysis/segment/loyalty_tier_funnel_chart.jpg", dpi=300)
plt.show()

# Tier-wise summary table
behavior_columns = ['RTO%', 'RTV%', 'Gross Sold Qty', 'Return_qty', 'Order Frequency',
                    'Lifespan', 'AOV', 'Discount%', 'Gross Profit']

# Mean and Median Summary
tier_summary_mean = df.groupby('Customer_Tier_Label')[behavior_columns].mean().round(2)
tier_summary_median = df.groupby('Customer_Tier_Label')[behavior_columns].median().round(2)

# Display summary
print("\nTier-wise Mean Summary:")
print(tier_summary_mean)

print("\nTier-wise Median Summary:")
print(tier_summary_median)


# %%
# Select key columns to export
final_columns = [
    'CUSTOMER_MOBILE', 'Customer_Cluster', 'Customer_Tier_Label', 'Customer_Score',
    'Order Frequency', 'AOV', 'Lifespan', 'RTO%', 'RTV%'
]

df_export = df1[final_columns]

output_path = "/Users/rarerabbit/Documents/Data Analysis/final_customer_segmentation.csv"
df_export.to_csv(output_path, index=False)

print(f"Final segmentation file saved to: {output_path}")


# %%



