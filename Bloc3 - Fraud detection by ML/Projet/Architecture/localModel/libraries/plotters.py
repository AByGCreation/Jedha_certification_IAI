# ==================== DATA MANIPULATION ====================
# ==================== VISUALIZATION ====================
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import ConfusionMatrixDisplay
from graphics import jedhaColor_black, jedhaColor_blue, jedhaColor_violet, jedha_bg_color, jedha_grid_color, jedha_font, jedhaCMInverted, jedhaCM
# ==================== DATA MANIPULATION ====================
import pandas as pd
import datetime
import folium
from folium.plugins import MarkerCluster
import os
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))

#======================================================================
## EDA - Plotting datas ##
#======================================================================
def plotFeatureDistributions(df: pd.DataFrame, current_path: str):
    """Plot distributions of numeric features for fraud and non-fraud cases.

    Args:
        df (pd.DataFrame): The preprocessed training DataFrame.
    """

    # Univariate analysis - Distribution of numeric variables
    print("Generating distributions for numeric features...")
    num_features = ["age", "amt", "trans_hour", "distance_km", "ccn_len"]

    for f in num_features:
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot non-fraud distribution
        # sns.histplot(
        #     data=df[df['is_fraud'] == 0],
        #     x=f,
        #     bins=50,
        #     label='Non-Fraude',
        #     color=jedhaColor_blue,
        #     alpha=0.6,
        #     ax=ax
        # )

        # Plot fraud distribution
        sns.histplot(
            data=df[df['is_fraud'] == 1],
            x=f,
            bins=50,
            label='Fraude',
            color=jedhaColor_violet,
            alpha=0.75,
            ax=ax
        )

        ax.set_title(f'Distribution - {f}', fontweight='bold', color=jedhaColor_black)
        ax.set_xlabel(f.capitalize(), color=jedhaColor_black)
        ax.set_ylabel('Frequency', color=jedhaColor_black)
        ax.set_facecolor(jedha_bg_color)
        fig.patch.set_facecolor(jedha_bg_color)
        ax.tick_params(colors=jedhaColor_black)
        ax.legend(facecolor=jedha_bg_color, edgecolor=jedhaColor_black)
        plt.tight_layout()

        plt.savefig(current_path + f"/outputs/Analysis_distribution_{f}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        plt.close(fig)

    print("✅ EDA analysis complete.")

    # Distribution of transaction amounts by fraud status

    print("Distribution des fraudes:")
    print(df['is_fraud'].value_counts())
    print()
    #print(df['is_fraud'].describe())
    print()

    # Visualize fraud distribution by category
    fig = go.Figure()

    # Fraud transactions by category
    fraud_by_category = df[df['is_fraud']==1].groupby('category').size().reset_index(name='count')
    fraud_by_category = fraud_by_category.sort_values('count', ascending=False)

    fig.add_trace(go.Bar(
        x=fraud_by_category['category'],
        y=fraud_by_category['count'],
        name='Transactions frauduleuses',
        marker_color=jedhaColor_violet
    ))

    fig.update_layout(
        title='Distribution des transactions frauduleuses par catégorie',
        xaxis_title='Catégorie',
        yaxis_title='Nombre de fraudes',
        height=500,
        width=1000
    )

    # Add Pareto curve (cumulative percentage)
    fraud_by_category['cumulative'] = fraud_by_category['count'].cumsum()
    fraud_by_category['cumulative_pct'] = fraud_by_category['cumulative'] / fraud_by_category['count'].sum() * 100

    fig.add_trace(go.Scatter(
        x=fraud_by_category['category'],
        y=fraud_by_category['cumulative_pct'],
        name='Courbe de Pareto (%)',
        mode='lines+markers',
        marker_color=jedhaColor_blue,
        yaxis='y2'
    ))

    fig.update_layout(
        yaxis2=dict(
            title='Pourcentage cumulatif (%)',
            overlaying='y',
            side='right',
            range=[0, 100],
            showgrid=False,
        )
    )

    fig.write_image(current_path + '/outputs/Analysis_transaction_fraud_distribution_by_category_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')

        
def drawCorrelationMatrix(df: pd.DataFrame, title_suffix: str = "", current_path: str = "") -> None:
    """Draw and save a correlation matrix heatmap for numeric variables in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        title_suffix (str, optional): Suffix to add to the title and filename. Defaults to "".
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=jedhaCMInverted, square=True)
    plt.title(f"Matrice de corrélation des variables numériques {title_suffix}")
    plt.savefig(current_path + f'/outputs/Analysis_correlationMatrix{title_suffix}_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
    plt.close()

def saveMap(df, nbPoint=None, outputPath=''):
    """Save a map with merchant locations and transaction clusters.

    Args:
        df (pd.DataFrame): DataFrame containing transaction data.
        nbPoint (int, optional): Number of points to plot. Defaults to None.
        outputPath (str, optional): Path to save the map HTML file. Defaults to ''.
    """
    # ~15min pour l'ensemble des points un fichier de 500mo
    
    # Center map on mean latitude and longitude of merchant locations
    center_lat = df['merch_lat'].astype(float).mean()
    center_lon = df['merch_long'].astype(float).mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles='CartoDB positron', control_scale=True, width='100%', height='100%', max_bounds=True)

    # Add merchant locations as points
    
    if nbPoint:
        dfTemp = df.head(nbPoint)
    else:
        dfTemp = df

    # Group by merchant and count number of transactions and frauds
    merchant_stats = dfTemp.groupby('merchant').agg(
        total_transactions=('is_fraud', 'size'),
        fraud_count=('is_fraud', 'sum')
    ).reset_index()

    # Draw points for merchant locations on the map
    # Create separate marker clusters for fraud and legitimate transactions

    fraud_cluster = MarkerCluster(name='Transactions frauduleuses').add_to(m)
    legit_cluster = MarkerCluster(name='Transactions légitimes').add_to(m)

    for idx, row in dfTemp.iterrows():
        lat = float(row['merch_lat'])
        lon = float(row['merch_long'])
        merchant = row['merchant']
        total_tx = merchant_stats.loc[merchant_stats['merchant'] == merchant, 'total_transactions'].values[0]
        fraud_tx = merchant_stats.loc[merchant_stats['merchant'] == merchant, 'fraud_count'].values[0]
        popup_text = (
            f"<b>Vendeur</b>: {merchant}<br>"
            f"<b>Montant</b>: {row['amt']}$ <br>"
            f"<b>Fraude</b>: {row['is_fraud']}<br>"
            f"<b>Nombre total de transactions</b>: {total_tx}<br>"
            f"<b>Nombre de transactions frauduleuses</b>: {fraud_tx}"
        )
        if row['is_fraud'] == 1:
            icon = folium.Icon(color='purple', icon='exclamation-sign', prefix='glyphicon')
            folium.Marker(
                location=[lat, lon],
                popup=popup_text,
                icon=icon
            ).add_to(fraud_cluster)
        else:
            icon = folium.Icon(color='lightblue', icon='ok-sign', prefix='glyphicon')
            folium.Marker(
                location=[lat, lon],
                popup=popup_text,
                icon=icon
            ).add_to(legit_cluster)

    # Add layer control to toggle clusters
    folium.LayerControl().add_to(m)

    # Add legend to the map
    legend_html = f'''
     <div id="customLegend" style="
         position: fixed; 
         bottom: 50px; left: 50px; width: 200px; height: 90px; 
         background-color: white; z-index:9999; font-size:14px;
         border:2px solid grey; border-radius:8px; padding: 10px;">
         <b>Légende</b><br>
         <i class="glyphicon glyphicon-exclamation-sign" style="color:{jedhaColor_violet}"></i> Transaction frauduleuse<br>
         <i class="glyphicon glyphicon-ok-sign" style="color:{jedhaColor_blue}"></i> Transaction légitime
     </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))


    m.save(outputPath, close_file=False)
    print(f"✅ Map saved to {outputPath}")


def draw_confusion_matrices(model_name, y_train, y_train_pred, y_test, y_test_pred):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training confusion matrix
    disp1 = ConfusionMatrixDisplay.from_predictions(
        y_train, y_train_pred, ax=ax1, cmap=jedhaCM, values_format='d'
    )

    ax1.set_title(f"{model_name} - Training Set", color=jedhaColor_black, fontsize=12, fontweight='bold', fontname={jedha_font})
    ax1.set_facecolor(jedha_bg_color)
    ax1.xaxis.label.set_color(jedhaColor_black)
    ax1.yaxis.label.set_color(jedhaColor_black)
    ax1.set_xticklabels(['Pas Fraude', 'Fraude'])
    ax1.set_yticklabels(['Pas Fraude', 'Fraude'])
    ax1.tick_params(colors=jedhaColor_black)

    # Test confusion matrix cx
    disp2 = ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred, ax=ax2, cmap=jedhaCM, values_format='d'
    )
    ax2.set_title(f"{model_name} - Test Set", color=jedhaColor_black, fontsize=12, fontweight='bold')
    ax2.set_facecolor(jedha_bg_color)
    ax2.xaxis.label.set_color(jedhaColor_black)
    ax2.yaxis.label.set_color(jedhaColor_black)
    ax2.set_xticklabels(['Pas Fraude', 'Fraude'])
    ax2.set_yticklabels(['Pas Fraude', 'Fraude'])
    ax2.tick_params(colors=jedhaColor_black)
    
    fig.patch.set_facecolor(jedha_bg_color)
    plt.tight_layout()
    plt.savefig(current_path + '/outputs/Results_confusionMatrix_' + model_name + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
    plt.close(fig)




            # Visualize model comparison

def draw_model_comparison(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(jedha_bg_color)
    
    # 1. Accuracy Comparison (Train vs Test)
    ax = axes[0, 0]
    x = np.arange(len(results_df))
    width = 0.35
    
    ax.bar(x - width/2, results_df['Train Accuracy'], width, 
        label='Train', color=jedhaColor_blue, alpha=0.8)
    ax.bar(x + width/2, results_df['Test Accuracy'], width, 
        label='Test', color=jedhaColor_violet, alpha=0.8)
    
    ax.set_xlabel('Model', fontweight='bold', color=jedhaColor_black)
    ax.set_ylabel('Accuracy', fontweight='bold', color=jedhaColor_black)
    ax.set_title('Accuracy Comparison: Train vs Test', fontweight='bold', color=jedhaColor_black)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
    ax.legend(facecolor=jedha_bg_color, edgecolor=jedhaColor_black)
    ax.set_facecolor(jedha_bg_color)
    ax.tick_params(colors=jedhaColor_black)
    ax.grid(True, alpha=0.3, color=jedhaColor_black)
    for spine in ax.spines.values():
        spine.set_color(jedhaColor_black)
    
    # 2. F1 Score Comparison (Train vs Test)
    ax = axes[0, 1]
    ax.bar(x - width/2, results_df['Train F1'], width, 
        label='Train', color=jedhaColor_blue, alpha=0.8)
    ax.bar(x + width/2, results_df['Test F1'], width, 
        label='Test', color=jedhaColor_violet, alpha=0.8)
    
    ax.set_xlabel('Model', fontweight='bold', color=jedhaColor_black)
    ax.set_ylabel('F1 Score', fontweight='bold', color=jedhaColor_black)
    ax.set_title('F1 Score Comparison: Train vs Test', fontweight='bold', color=jedhaColor_black)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
    ax.legend(facecolor=jedha_bg_color, edgecolor=jedhaColor_black)
    ax.set_facecolor(jedha_bg_color)
    ax.tick_params(colors=jedhaColor_black)
    ax.grid(True, alpha=0.3, color=jedhaColor_black)
    for spine in ax.spines.values():
        spine.set_color(jedhaColor_black)
    
    # 3. Overfitting Detection (Accuracy Gap)
    # ax = axes[1, 0]
    # accuracy_gap = results_df['Train Accuracy'] - results_df['Test Accuracy']
    # colors_gap = [jedhaColor_violet if gap > 0.05 else jedhaColor_blue for gap in accuracy_gap]
    
    # ax.bar(results_df['Model'], accuracy_gap, color=colors_gap, alpha=0.8)
    # ax.axhline(y=0, color=jedhaColor_black, linestyle='-', linewidth=0.5)
    # ax.axhline(y=0.01, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overfitting threshold')
    
    # ax.set_xlabel('Model', fontweight='bold', color=jedhaColor_black)
    # ax.set_ylabel('Train-Test Accuracy', fontweight='bold', color=jedhaColor_black)
    # ax.set_title('Detection Overfitting ', fontweight='bold', color=jedhaColor_black)
    # ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
    # ax.legend(facecolor=jedha_bg_color, edgecolor=jedhaColor_black)
    # ax.set_facecolor(jedha_bg_color)
    # ax.tick_params(colors=jedhaColor_black)
    # ax.grid(True, alpha=0.3, color=jedhaColor_black)
    # for spine in ax.spines.values():
    #     spine.set_color(jedhaColor_black)
    
    # 3. ROC-AUC Comparison (Train vs Test)
    ax = axes[1, 0]
    if 'Train ROC AUC' in results_df.columns and 'Test ROC AUC' in results_df.columns:
        ax.bar(x - width/2, results_df['Train ROC AUC'], width,
            label='Train', color=jedhaColor_blue, alpha=0.8)
        ax.bar(x + width/2, results_df['Test ROC AUC'], width,
            label='Test', color=jedhaColor_violet, alpha=0.8)

        ax.set_xlabel('Model', fontweight='bold', color=jedhaColor_black)
        ax.set_ylabel('ROC AUC Score', fontweight='bold', color=jedhaColor_black)
        ax.set_title('ROC AUC Comparison: Train vs Test', fontweight='bold', color=jedhaColor_black)
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
        ax.legend(facecolor=jedha_bg_color, edgecolor=jedhaColor_black)
        ax.set_facecolor(jedha_bg_color)
        ax.tick_params(colors=jedhaColor_black)
        ax.grid(True, alpha=0.3, color=jedhaColor_black)
        for spine in ax.spines.values():
            spine.set_color(jedhaColor_black)
    else:
        ax.text(0.5, 0.5, 'ROC AUC data not available',
                ha='center', va='center', fontsize=12, color=jedhaColor_black)
        ax.set_facecolor(jedha_bg_color)
        for spine in ax.spines.values():
            spine.set_color(jedhaColor_black)

    # 4. Training Time Comparison
    ax = axes[1, 1]
    ax.bar(results_df['Model'], results_df['Time (s)'], color=jedhaColor_blue, alpha=0.8)
    
    ax.set_xlabel('Model', fontweight='bold', color=jedhaColor_black)
    ax.set_ylabel('Training Time (seconds)', fontweight='bold', color=jedhaColor_black)
    ax.set_title('Training Time Comparison', fontweight='bold', color=jedhaColor_black)
    ax.set_xticklabels(results_df['Model'], rotation=15, ha='right')
    ax.set_facecolor(jedha_bg_color)
    ax.tick_params(colors=jedhaColor_black)
    ax.grid(True, alpha=0.3, color=jedhaColor_black)
    for spine in ax.spines.values():
        spine.set_color(jedhaColor_black)
    
    plt.suptitle('Comparaison des modeles',
                fontsize=16, fontweight='bold', color=jedhaColor_black, y=0.995)
    plt.tight_layout()
    plt.savefig(current_path + '/outputs/Results_model_performance_comparison_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.pdf')
    plt.close(fig)
            