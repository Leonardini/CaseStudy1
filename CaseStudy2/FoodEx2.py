# Import necessary libraries  
import pandas as pd  
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn.cluster import KMeans  
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import collections

# Read the Excel file (Sheet1)
FILEPATH = 'FoodEx2-CaseStudy2-Dataset_V1.xlsx'
xlsx = pd.ExcelFile(FILEPATH, engine='openpyxl')  
df = pd.read_excel(xlsx, sheet_name='Sheet1')
print(df.head())
  
# Load the SentenceTransformer model (using a light model for speed)
model = SentenceTransformer('all-MiniLM-L6-v2')  
  
# Generate embeddings for each input row, using ENFOODNAME as the input
embeddings = model.encode(df['ENFOODNAME'].astype(str).tolist(), show_progress_bar=True)
  
# Cluster the embeddings using k-means:  
n_clusters = 3  # assume 3 clusters for the moment
kmeans = KMeans(n_clusters = n_clusters, random_state = 42)
clusters = kmeans.fit_predict(embeddings)  
df['cluster'] = clusters  
# Cluster distribution
print(df['cluster'].value_counts())

# Standardize embeddings
scaler = StandardScaler()
scaled_embeddings = scaler.fit_transform(embeddings)

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_embeddings)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
# Add names (label 1)
if 'BASETERM_NAME' in df.columns:
    pca_df['food_name'] = df['BASETERM_NAME'].values
# Add facets (label 2)
if 'FACETS' in df.columns:
    pca_df['category'] = df['FACETS'].values

# Calculate the explained variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: {explained_variance}")
print(f"Total explained variance: {sum(explained_variance):.2f}")

# Plot the PCA results
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

# If we have many categories, limit to top N for clarity
if pca_df['category'].nunique() > 10:
    top_categories = pca_df['category'].value_counts().nlargest(10).index
    plot_df = pca_df[pca_df['category'].isin(top_categories)].copy()
    title_suffix = " (Top 10 Categories)"
else:
    plot_df = pca_df.copy()
    title_suffix = ""

# Create the scatter plot
scatter = sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='category',
    palette='viridis',
    alpha=0.7,
    s=100,
    data=plot_df
)

# Improve the plot appearance
plt.title(f'PCA of Food Embeddings{title_suffix}', fontsize=20, pad=15)
plt.xlabel('Principal Component 1', fontsize=16, labelpad=10)
plt.ylabel('Principal Component 2', fontsize=16, labelpad=10)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='Category', fontsize=12, title_fontsize=14)
plt.grid(True, alpha=0.3)

# Add annotations for some points (limit to avoid clutter)
np.random.seed(42)
sample_indices = np.random.choice(len(plot_df), min(20, len(plot_df)), replace=False)
for i in sample_indices:
    plt.annotate(
        plot_df.iloc[i]['food_name'][:15] + ('...' if len(plot_df.iloc[i]['food_name']) > 15 else ''),
        (plot_df.iloc[i]['PC1'], plot_df.iloc[i]['PC2']),
        fontsize=9,
        alpha=0.7,
        xytext=(5, 5),
        textcoords='offset points'
    )

plt.tight_layout()
plt.show()

# Let us also create a 3D PCA visualization
pca_3d = PCA(n_components=3)
principal_components_3d = pca_3d.fit_transform(scaled_embeddings)

# Create a DataFrame with the 3D principal components
pca_3d_df = pd.DataFrame(data=principal_components_3d, columns=['PC1', 'PC2', 'PC3'])

# Add the same metadata: names and facets
if 'BASETERM_NAME' in df.columns:
    pca_3d_df['food_name'] = df['BASETERM_NAME'].values

if 'FACETS' in df.columns:
    pca_3d_df['category'] = df['FACETS'].values

# Calculate the 3D explained variance
explained_variance_3d = pca_3d.explained_variance_ratio_
print(f"3D PCA explained variance ratio: {explained_variance_3d}")
print(f"Total 3D explained variance: {sum(explained_variance_3d):.2f}")

# Create a 3D plot
from mpl_toolkits.mplot3d import Axes3D

# If we have many categories, limit to top 10 for clarity
if pca_3d_df['category'].nunique() > 10:
    top_categories_3d = pca_3d_df['category'].value_counts().nlargest(10).index
    plot_3d_df = pca_3d_df[pca_3d_df['category'].isin(top_categories_3d)].copy()
    title_suffix_3d = " (Top 10 Categories)"
else:
    plot_3d_df = pca_3d_df.copy()
    title_suffix_3d = ""

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Get unique categories and assign colors
categories = plot_3d_df['category'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))

# Plot each category with a different color
for i, category in enumerate(categories):
    category_data = plot_3d_df[plot_3d_df['category'] == category]
    ax.scatter(
        category_data['PC1'],
        category_data['PC2'],
        category_data['PC3'],
        color=colors[i],
        s=50,
        alpha=0.7,
        label=category
    )

# Set labels and title
ax.set_xlabel('Principal Component 1', fontsize=14, labelpad=10)
ax.set_ylabel('Principal Component 2', fontsize=14, labelpad=10)
ax.set_zlabel('Principal Component 3', fontsize=14, labelpad=10)
ax.set_title(f'3D PCA of Food Embeddings{title_suffix_3d}', fontsize=18, pad=15)

# Add a legend
ax.legend(title='Category', fontsize=12)

# Adjust the viewing angle for better visualization
ax.view_init(elev=30, azim=45)

plt.tight_layout()
plt.show()

# Rename the dataset
food_data = df
# Drop rows with missing FACETS
food_data = food_data.dropna(subset=['FACETS'])
# Drop corresponding embeddings as well
check_nan = df['FACETS'].isnull().values
embeddings = [x for (i,x) in enumerate(embeddings) if not check_nan[i]]

# Extract the top-level label (before the first period)
food_data['top_level'] = food_data['FACETS'].apply(
    lambda x: str(x).split('.')[0] if '.' in str(x) else str(x).split('#')[0] if '#' in str(x) else str(x)
)

# Check the distribution of top-level labels
top_level_counts = food_data['top_level'].value_counts()
print("Number of unique top-level labels:", len(top_level_counts))
print("\
Top 10 most common top-level labels:")
print(top_level_counts.head(10))

np.random.seed(42)

# Count occurrences of each top-level label
top_level_counts = food_data['top_level'].value_counts()
print("Number of unique top-level labels:", len(top_level_counts))
# Filter to keep only classes with at least 10 samples
min_samples = 10
valid_classes = top_level_counts[top_level_counts >= min_samples].index.tolist()
print(f"Number of classes with at least {min_samples} samples:", len(valid_classes))
badRows = [i for (i,x) in enumerate(food_data['top_level'].isin(valid_classes)) if not x]
embeddings = [embeddings[x] for x in range(len(embeddings)) if x not in badRows]
food_data = food_data.reset_index(drop=True)
food_data = food_data.drop(badRows)
# Encode the target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(food_data['top_level'])

# Plot the distribution of the filtered classes
plt.figure(figsize=(12, 8))
sns.barplot(x=top_level_counts[valid_classes].values[:15],
            y=top_level_counts[valid_classes].index[:15])
plt.title('Top 15 Food Categories', fontsize=20, pad=15)
plt.xlabel('Count', fontsize=16, labelpad=10)
plt.ylabel('Category', fontsize=16, labelpad=10)
plt.tight_layout()
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, y, test_size=0.2, random_state=42, stratify=y)

print("\
Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Number of classes:", len(label_encoder.classes_))

# Plot the distribution of classes in the training set
plt.figure(figsize=(12, 8))
sns.countplot(y=y_train, order=np.bincount(y_train).argsort()[::-1][:20])
plt.title('Distribution of Top 20 Classes in Training Set', fontsize=20, pad=15)
plt.xlabel('Count', fontsize=16, labelpad=10)
plt.ylabel('Class Index', fontsize=16, labelpad=10)
plt.tight_layout()
plt.show()

# Map the class indices back to their original labels for the top 10 classes
top_10_indices = np.bincount(y_train).argsort()[::-1][:10]
top_10_labels = [label_encoder.inverse_transform([idx])[0] for idx in top_10_indices]
top_10_counts = [np.bincount(y_train)[idx] for idx in top_10_indices]

print("\
Top 10 classes with their original labels:")
for i, (label, count) in enumerate(zip(top_10_labels, top_10_counts)):
    print(f"{i+1}. {label}: {count} samples")

# Now let's create features from the food names using TF-IDF
print("\
Creating TF-IDF features from food names...")
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X = tfidf.fit_transform(food_data['ENFOODNAME'].fillna(''))

# Encode the target labels
y = label_encoder.fit_transform(food_data['top_level'])

# Print the mapping of encoded labels to original labels for reference
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
print("\
Label mapping (first 10):")
for i, (idx, label) in enumerate(list(label_mapping.items())[:10]):
    print(f"{idx}: {label}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\
Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
print("Number of classes:", len(label_encoder.classes_))

# Initialize the classifier with some reasonable parameters
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\
Accuracy: {accuracy:.4f}")

# Print classification report for the top 10 most frequent classes
top_classes = top_level_counts[valid_classes].nlargest(10).index
top_class_indices = [list(label_encoder.classes_).index(cls) for cls in top_classes]

# Map indices back to class names for better readability
target_names = [label_mapping[i] for i in range(len(label_encoder.classes_))]

# Print classification report
print("\
Classification Report:")
print(classification_report(y_test, y_pred, labels=top_class_indices,
                           target_names=[target_names[i] for i in top_class_indices]))

# Plot feature importance
feature_importances = pd.DataFrame({
    'feature': tfidf.get_feature_names_out(),
    'importance': rf.feature_importances_
})
feature_importances = feature_importances.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
plt.title('Top 20 Most Important Features', fontsize=20, pad=15)
plt.xlabel('Importance', fontsize=16, labelpad=10)
plt.ylabel('Feature', fontsize=16, labelpad=10)
plt.tight_layout()
plt.show()

# Plot confusion matrix for the top 5 classes
top5_indices = top_class_indices[:5]
top5_names = [target_names[i] for i in top5_indices]

# Filter test and prediction data for top 5 classes
mask_test = np.isin(y_test, top5_indices)
y_test_top5 = y_test[mask_test]
y_pred_top5 = y_pred[mask_test]

# Create confusion matrix
cm = confusion_matrix(y_test_top5, y_pred_top5, labels=top5_indices)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=top5_names, yticklabels=top5_names)
plt.title('Confusion Matrix for Top 5 Classes', fontsize=20, pad=15)
plt.xlabel('Predicted', fontsize=16, labelpad=10)
plt.ylabel('True', fontsize=16, labelpad=10)
plt.tight_layout()
plt.show()

# Let's also try feature selection to improve the model
print("\
Performing feature selection...")
selector = SelectFromModel(rf, threshold='median')
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

print(f"Number of features before selection: {X_train.shape[1]}")
print(f"Number of features after selection: {X_train_selected.shape[1]}")

# Train a new model with selected features
rf_selected = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_selected.fit(X_train_selected, y_train)
y_pred_selected = rf_selected.predict(X_test_selected)

# Evaluate the model with selected features
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print(f"\
Accuracy with feature selection: {accuracy_selected:.4f}")

# Compare the two models
print("\
Model comparison:")
print(f"Original model accuracy: {accuracy:.4f}")
print(f"Feature-selected model accuracy: {accuracy_selected:.4f}")
print(f"Difference: {accuracy_selected - accuracy:.4f}")

filtered_df = food_data

# We will use the 'ENFOODNAME' column for creating sentence embeddings
model_name = 'all-MiniLM-L6-v2'
print("Loading SentenceTransformer model (" + model_name + ")...")
embedder = SentenceTransformer(model_name)

# Get the food descriptions
descriptions = filtered_df['ENFOODNAME'].astype(str).tolist()
start_time = time.time()
embeddings = embedder.encode(descriptions, show_progress_bar=True)
print("Embeddings computed in {:.2f} seconds".format(time.time() - start_time))

X = embeddings
le = LabelEncoder()
y = le.fit_transform(filtered_df['top_level'])

print("Number of top-level classes:", len(le.classes_))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Train a Random Forest classifier
print("Training Random Forest classifier on sentence embeddings...")
rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
start_time = time.time()
rf.fit(X_train, y_train)
print("Training completed in {:.2f} seconds".format(time.time() - start_time))

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set: {:.4f}".format(accuracy))

# Output classification report
print("\
Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Plot confusion matrix for the top 5 most frequent top-level classes
frequent_classes = top_level_counts.loc[valid_classes].nlargest(5).index
frequent_class_indices = [np.where(le.classes_ == cls)[0][0] for cls in frequent_classes]
mask = np.isin(y_test, frequent_class_indices)
y_test_top5 = y_test[mask]
y_pred_top5 = y_pred[mask]
cm = confusion_matrix(y_test_top5, y_pred_top5, labels=frequent_class_indices)

plt.figure(figsize=(9, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=frequent_classes, yticklabels=frequent_classes)
plt.title('Confusion Matrix for Top 5 Top-Level Classes', fontsize=20, pad=15)
plt.xlabel('Predicted', fontsize=16, labelpad=10)
plt.ylabel('True', fontsize=16, labelpad=10)
plt.tight_layout()
plt.show()