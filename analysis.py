import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import os

# ── Setup ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)
DATA_PATH = "data/Social Media Engagement Dataset.csv"

df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

# ── 1  Data Preprocessing ─────────────────────────────────────────────
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["has_mentions"] = df["mentions"].notna().astype(int)
df["total_interactions"] = df["likes_count"] + df["shares_count"] + df["comments_count"]
df["hashtag_count"] = df["hashtags"].str.count(",") + 1

# Numeric columns for correlation
NUM_COLS = [
    "sentiment_score", "toxicity_score", "likes_count", "shares_count",
    "comments_count", "impressions", "engagement_rate",
    "user_past_sentiment_avg", "user_engagement_growth", "buzz_change_rate",
    "total_interactions", "hashtag_count",
]

print("Preprocessing done.\n")
print(df[NUM_COLS].describe().round(3).to_string())

# 2  Distribution Plots

# 2a  Engagement rate distribution (log-transformed)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
axes[0].hist(df["engagement_rate"], bins=80, color="steelblue", edgecolor="white")
axes[0].set_title("Engagement Rate Distribution")
axes[0].set_xlabel("Engagement Rate")
axes[0].set_ylabel("Frequency")

axes[1].hist(df["sentiment_score"], bins=60, color="coral", edgecolor="white")
axes[1].set_title("Sentiment Score Distribution")
axes[1].set_xlabel("Sentiment Score")

axes[2].hist(df["toxicity_score"], bins=60, color="mediumpurple", edgecolor="white")
axes[2].set_title("Toxicity Score Distribution")
axes[2].set_xlabel("Toxicity Score")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_distributions.pdf"), dpi=150)
plt.close()
print("✓ fig1_distributions.pdf")

# 2b  Box plots: engagement metrics by platform
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col, color in zip(
    axes,
    ["likes_count", "shares_count", "comments_count"],
    ["#4C72B0", "#55A868", "#C44E52"],
):
    sns.boxplot(data=df, x="platform", y=col, ax=ax, color=color, fliersize=2)
    ax.set_title(f"{col.replace('_', ' ').title()} by Platform")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_platform_boxplots.pdf"), dpi=150)
plt.close()
print("✓ fig2_platform_boxplots.pdf")

#  3  Correlation Heatmap
corr = df[NUM_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
    center=0, linewidths=0.5, ax=ax, square=True, vmin=-1, vmax=1
)
ax.set_title("Pearson Correlation Matrix", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_correlation_heatmap.pdf"), dpi=150)
plt.close()
print("✓ fig3_correlation_heatmap.pdf")

# Print top correlations
upper = corr.where(mask == False)
corr_pairs = (
    upper.stack().reset_index()
    .rename(columns={"level_0": "Var1", "level_1": "Var2", 0: "r"})
)
corr_pairs["abs_r"] = corr_pairs["r"].abs()
print("\nTop 10 correlations:")
print(corr_pairs.nlargest(10, "abs_r")[["Var1", "Var2", "r"]].to_string(index=False))

# 4  Sentiment vs Engagement Scatter
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(df["sentiment_score"], df["engagement_rate"],
                alpha=0.15, s=8, c="steelblue")
axes[0].set_xlabel("Sentiment Score ($s$)")
axes[0].set_ylabel("Engagement Rate ($E$)")
axes[0].set_title("Sentiment vs Engagement Rate")
# Add LOWESS trend
from statsmodels.nonparametric.smoothers_lowess import lowess
try:
    lw = lowess(df["engagement_rate"], df["sentiment_score"], frac=0.3)
    axes[0].plot(lw[:, 0], lw[:, 1], color="red", lw=2, label="LOWESS")
    axes[0].legend()
except Exception:
    pass

axes[1].scatter(df["toxicity_score"], df["engagement_rate"],
                alpha=0.15, s=8, c="coral")
axes[1].set_xlabel("Toxicity Score ($\\tau$)")
axes[1].set_ylabel("Engagement Rate ($E$)")
axes[1].set_title("Toxicity vs Engagement Rate")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_scatter_sentiment_engagement.pdf"), dpi=150)
plt.close()
print("✓ fig4_scatter_sentiment_engagement.pdf")

#  5  Temporal Patterns
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By hour
hourly = df.groupby("hour")["engagement_rate"].agg(["mean", "std"]).reset_index()
axes[0].errorbar(hourly["hour"], hourly["mean"], yerr=hourly["std"] / np.sqrt(len(df)),
                 fmt="o-", color="steelblue", capsize=3)
axes[0].set_xlabel("Hour of Day ($h$)")
axes[0].set_ylabel("Mean Engagement Rate")
axes[0].set_title("Engagement Rate by Hour")
axes[0].set_xticks(range(0, 24, 2))

# By day of week
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
daily = df.groupby("day_of_week")["engagement_rate"].mean().reindex(day_order)
axes[1].bar(range(7), daily.values, color=sns.color_palette("viridis", 7))
axes[1].set_xticks(range(7))
axes[1].set_xticklabels([d[:3] for d in day_order])
axes[1].set_xlabel("Day of Week")
axes[1].set_ylabel("Mean Engagement Rate")
axes[1].set_title("Engagement Rate by Day of Week")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig5_temporal_patterns.pdf"), dpi=150)
plt.close()
print("✓ fig5_temporal_patterns.pdf")

#  6  Brand & Campaign Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Brand engagement
brand_eng = df.groupby("brand_name")["engagement_rate"].mean().sort_values(ascending=True)
axes[0].barh(brand_eng.index, brand_eng.values, color="steelblue")
axes[0].set_xlabel("Mean Engagement Rate")
axes[0].set_title("Mean Engagement Rate by Brand")

# Campaign phase
phase_order = ["Pre-Launch", "Launch", "Post-Launch"]
phase_stats = df.groupby("campaign_phase")[["engagement_rate", "sentiment_score"]].mean().reindex(phase_order)
x = np.arange(len(phase_order))
w = 0.35
axes[1].bar(x - w/2, phase_stats["engagement_rate"], w, label="Engagement Rate", color="steelblue")
ax2 = axes[1].twinx()
ax2.bar(x + w/2, phase_stats["sentiment_score"], w, label="Sentiment Score", color="coral")
axes[1].set_xticks(x)
axes[1].set_xticklabels(phase_order)
axes[1].set_ylabel("Engagement Rate", color="steelblue")
ax2.set_ylabel("Sentiment Score", color="coral")
axes[1].set_title("Metrics by Campaign Phase")
axes[1].legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig6_brand_campaign.pdf"), dpi=150)
plt.close()
print("✓ fig6_brand_campaign.pdf")

#  7  Emotion & Sentiment Interaction
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Engagement by emotion type
sns.boxplot(data=df, x="emotion_type", y="engagement_rate", ax=axes[0],
            palette="Set2", fliersize=2,
            order=["Happy", "Excited", "Confused", "Sad", "Angry"])
axes[0].set_title("Engagement Rate by Emotion Type")
axes[0].set_xlabel("Emotion Type")
axes[0].set_ylabel("Engagement Rate")

# Sentiment label × Platform heatmap
pivot = df.pivot_table(values="engagement_rate", index="platform",
                       columns="sentiment_label", aggfunc="mean")
sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=axes[1])
axes[1].set_title("Mean Engagement: Platform × Sentiment")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig7_emotion_sentiment.pdf"), dpi=150)
plt.close()
print("✓ fig7_emotion_sentiment.pdf")

#  8  Topic Category Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

topic_eng = df.groupby("topic_category")["engagement_rate"].mean().sort_values(ascending=True)
axes[0].barh(topic_eng.index, topic_eng.values, color="teal")
axes[0].set_xlabel("Mean Engagement Rate")
axes[0].set_title("Engagement Rate by Topic Category")

# Topic × sentiment heatmap
pivot2 = df.pivot_table(values="engagement_rate", index="topic_category",
                        columns="sentiment_label", aggfunc="mean")
sns.heatmap(pivot2, annot=True, fmt=".3f", cmap="coolwarm", ax=axes[1])
axes[1].set_title("Engagement: Topic × Sentiment")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig8_topic_analysis.pdf"), dpi=150)
plt.close()
print("✓ fig8_topic_analysis.pdf")

#  9  PCA Visualization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[NUM_COLS].dropna())
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=df.loc[df[NUM_COLS].dropna().index, "sentiment_score"],
                     cmap="RdYlGn", alpha=0.3, s=5)
plt.colorbar(scatter, label="Sentiment Score")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA of Engagement Features (colored by Sentiment)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig9_pca.pdf"), dpi=150)
plt.close()
print("✓ fig9_pca.pdf")

# Print PCA loadings
print("\nPCA Loadings (PC1, PC2):")
for name, l1, l2 in zip(NUM_COLS, pca.components_[0], pca.components_[1]):
    print(f"  {name:30s}  {l1:+.3f}  {l2:+.3f}")
print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")

# 10  Regression: Feature Importance
features = [
    "sentiment_score", "toxicity_score", "impressions",
    "user_past_sentiment_avg", "user_engagement_growth",
    "buzz_change_rate", "hashtag_count", "has_mentions", "hour",
]
X = df[features].values
y = df["engagement_rate"].values

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)
importances = rf.feature_importances_

fig, ax = plt.subplots(figsize=(8, 5))
idx = np.argsort(importances)
ax.barh([features[i] for i in idx], importances[idx], color="steelblue")
ax.set_xlabel("Feature Importance (MDI)")
ax.set_title("Random Forest Feature Importance for Engagement Rate")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig10_feature_importance.pdf"), dpi=150)
plt.close()
print("✓ fig10_feature_importance.pdf")

# Cross-validation scores
lr = LinearRegression()
lr_scores = cross_val_score(lr, X, y, cv=5, scoring="r2")
rf_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
print(f"\nLinear Regression CV R²: {lr_scores.mean():.4f} ± {lr_scores.std():.4f}")
print(f"Random Forest CV R²:    {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

#  11  Statistical Tests
print("\n── Statistical Tests ──")

# ANOVA: engagement_rate across platforms
groups = [g["engagement_rate"].values for _, g in df.groupby("platform")]
f_stat, p_val = stats.f_oneway(*groups)
print(f"ANOVA (engagement_rate ~ platform): F={f_stat:.4f}, p={p_val:.4e}")

# ANOVA: engagement_rate across sentiment labels
groups2 = [g["engagement_rate"].values for _, g in df.groupby("sentiment_label")]
f_stat2, p_val2 = stats.f_oneway(*groups2)
print(f"ANOVA (engagement_rate ~ sentiment_label): F={f_stat2:.4f}, p={p_val2:.4e}")

# Pearson correlation: sentiment vs engagement
r_se, p_se = stats.pearsonr(df["sentiment_score"], df["engagement_rate"])
print(f"Pearson r(sentiment, engagement): r={r_se:.4f}, p={p_se:.4e}")

# Pearson correlation: toxicity vs engagement
r_te, p_te = stats.pearsonr(df["toxicity_score"], df["engagement_rate"])
print(f"Pearson r(toxicity, engagement):  r={r_te:.4f}, p={p_te:.4e}")

# Spearman: buzz_change_rate vs engagement
r_be, p_be = stats.spearmanr(df["buzz_change_rate"], df["engagement_rate"])
print(f"Spearman ρ(buzz_change, engagement): ρ={r_be:.4f}, p={p_be:.4e}")

# Chi-square: sentiment_label vs emotion_type
ct = pd.crosstab(df["sentiment_label"], df["emotion_type"])
chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
print(f"Chi² (sentiment_label × emotion_type): χ²={chi2:.2f}, dof={dof}, p={p_chi:.4e}")

# ── 12  Buzz Change Rate Distribution ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df["buzz_change_rate"], bins=60, color="teal", edgecolor="white", density=True)
# Overlay normal fit
mu, sigma = df["buzz_change_rate"].mean(), df["buzz_change_rate"].std()
x_range = np.linspace(df["buzz_change_rate"].min(), df["buzz_change_rate"].max(), 200)
axes[0].plot(x_range, stats.norm.pdf(x_range, mu, sigma), "r-", lw=2, label=f"N({mu:.1f}, {sigma:.1f}²)")
axes[0].set_title("Buzz Change Rate Distribution")
axes[0].set_xlabel("Buzz Change Rate ($\\Delta B$)")
axes[0].set_ylabel("Density")
axes[0].legend()

# QQ plot
stats.probplot(df["buzz_change_rate"], dist="norm", plot=axes[1])
axes[1].set_title("Q-Q Plot: Buzz Change Rate")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig11_buzz_distribution.pdf"), dpi=150)
plt.close()
print("✓ fig11_buzz_distribution.pdf")

# Shapiro-Wilk on sample
sample = df["buzz_change_rate"].sample(5000, random_state=42)
sw_stat, sw_p = stats.shapiro(sample)
print(f"Shapiro-Wilk (buzz_change_rate, n=5000): W={sw_stat:.4f}, p={sw_p:.4e}")

print("\n All figures saved to:", FIG_DIR)
