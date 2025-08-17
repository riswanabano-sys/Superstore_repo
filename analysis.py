#!/usr/bin/env python3
"""
Reproducible EDA + Profit Prediction for Tableau Superstore dataset.
Outputs: reports/Superstore_Analysis_Report.pdf
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import textwrap

DATA_PATH = os.path.join('data', 'Sample - Superstore.csv')
REPORT_PATH = os.path.join('reports', 'Superstore_Analysis_Report.pdf')

def main():
    # Load
    df = pd.read_csv(DATA_PATH, encoding='latin1')
    df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], errors='coerce')
    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.to_period('M').astype(str)
    df['Ship Delay (days)'] = (df['Ship Date'] - df['Order Date']).dt.days
    df['Profit Margin'] = np.where(df['Sales']!=0, df['Profit']/df['Sales'], np.nan)

    # Aggregations
    monthly = df.groupby('Month', as_index=False)[['Sales','Profit']].sum()
    monthly['Month_dt'] = pd.to_datetime(monthly['Month'], format='%Y-%m')
    monthly = monthly.sort_values('Month_dt')

    by_category = df.groupby('Category', as_index=False).agg(
        Sales=('Sales','sum'), Profit=('Profit','sum'), Orders=('Order ID','nunique'))
    by_category['Profit Margin'] = np.where(by_category['Sales']>0, by_category['Profit']/by_category['Sales'], np.nan)

    by_subcat = df.groupby('Sub-Category', as_index=False).agg(
        Sales=('Sales','sum'), Profit=('Profit','sum'), Orders=('Order ID','nunique')).sort_values('Profit', ascending=False)
    by_subcat['Profit Margin'] = np.where(by_subcat['Sales']>0, by_subcat['Profit']/by_subcat['Sales'], np.nan)

    by_region = df.groupby('Region', as_index=False).agg(Sales=('Sales','sum'), Profit=('Profit','sum'))
    by_region['Profit Margin'] = np.where(by_region['Sales']>0, by_region['Profit']/by_region['Sales'], np.nan)

    by_segment = df.groupby('Segment', as_index=False).agg(Sales=('Sales','sum'), Profit=('Profit','sum'))
    by_segment['Profit Margin'] = np.where(by_segment['Sales']>0, by_segment['Profit']/by_segment['Sales'], np.nan)

    by_shipmode = df.groupby('Ship Mode', as_index=False).agg(Sales=('Sales','sum'), Profit=('Profit','sum'))
    by_shipmode['Profit Margin'] = np.where(by_shipmode['Sales']>0, by_shipmode['Profit']/by_shipmode['Sales'], np.nan)

    bins = [-0.01, 0, 0.1, 0.2, 0.3, 0.4, 1.0]
    labels = ['0','0-10%','10-20%','20-30%','30-40%','40%+']
    df['Discount Bucket'] = pd.cut(df['Discount'], bins=bins, labels=labels)
    by_disc = df.groupby('Discount Bucket', as_index=False).agg(Sales=('Sales','sum'), Profit=('Profit','sum'))
    by_disc['Profit Margin'] = np.where(by_disc['Sales']>0, by_disc['Profit']/by_disc['Sales'], np.nan)

    # KPIs
    total_sales = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    overall_margin = (total_profit/total_sales) if total_sales else np.nan
    unique_customers = df['Customer ID'].nunique()
    unique_orders = df['Order ID'].nunique()

    # Modeling
    model_df = df.dropna(subset=['Profit','Sales','Quantity','Discount','Ship Delay (days)']).copy()
    features = ['Sales','Quantity','Discount','Ship Delay (days)','Category','Sub-Category','Segment','Region','Ship Mode']
    cat_cols = ['Category','Sub-Category','Segment','Region','Ship Mode']
    X = pd.get_dummies(model_df[features], columns=cat_cols, drop_first=True)
    y = model_df['Profit'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin = LinearRegression().fit(X_train, y_train)
    y_pred_lin = lin.predict(X_test)
    r2_lin = r2_score(y_test, y_pred_lin)
    rmse_lin = mean_squared_error(y_test, y_pred_lin, squared=False)
    mae_lin = mean_absolute_error(y_test, y_pred_lin)

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)

    fi = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=False)

    # Build PDF
    pp = PdfPages(REPORT_PATH)

    def add_text_page(title, lines):
        fig = plt.figure(figsize=(11.69,8.27))
        plt.axis('off')
        plt.text(0.02, 0.92, textwrap.fill(title, 80), fontsize=18, fontweight='bold', va='top')
        y = 0.86
        for line in lines:
            plt.text(0.03, y, textwrap.fill(line, 110), fontsize=12, va='top')
            y -= 0.06
            if y < 0.06:
                pp.savefig(fig, bbox_inches='tight'); plt.close(fig)
                fig = plt.figure(figsize=(11.69,8.27)); plt.axis('off'); y = 0.92
        pp.savefig(fig, bbox_inches='tight'); plt.close(fig)

    cover = [
        f"Total Sales: {total_sales:,.2f}", f"Total Profit: {total_profit:,.2f}",
        f"Overall Profit Margin: {overall_margin*100:.2f}%",
        f"Unique Orders: {unique_orders:,} | Unique Customers: {unique_customers:,}",
        "Dataset: Tableau Superstore (9,994 rows, 21 columns)"
    ]
    add_text_page("Superstore Sales & Profit Analysis — Executive Summary", cover)

    # Charts
    fig = plt.figure(figsize=(11.69,8.27)); plt.plot(monthly['Month_dt'], monthly['Sales']); plt.title('Monthly Sales'); plt.xlabel('Month'); plt.ylabel('Sales'); plt.xticks(rotation=45); plt.tight_layout(); pp.savefig(fig); plt.close(fig)
    fig = plt.figure(figsize=(11.69,8.27)); plt.plot(monthly['Month_dt'], monthly['Profit']); plt.title('Monthly Profit'); plt.xlabel('Month'); plt.ylabel('Profit'); plt.xticks(rotation=45); plt.tight_layout(); pp.savefig(fig); plt.close(fig)
    fig = plt.figure(figsize=(11.69,8.27)); plt.bar(by_category['Category'], by_category['Profit']); plt.title('Profit by Category'); plt.xlabel('Category'); plt.ylabel('Total Profit'); plt.tight_layout(); pp.savefig(fig); plt.close(fig)
    fig = plt.figure(figsize=(11.69,8.27)); top10 = by_subcat.head(10).sort_values('Profit', ascending=True); plt.barh(top10['Sub-Category'], top10['Profit']); plt.title('Top 10 Sub-Categories by Profit'); plt.xlabel('Profit'); plt.tight_layout(); pp.savefig(fig); plt.close(fig)
    fig = plt.figure(figsize=(11.69,8.27)); bottom10 = by_subcat.sort_values('Profit').head(10); plt.barh(bottom10['Sub-Category'], bottom10['Profit']); plt.title('Bottom 10 Sub-Categories (Loss Makers)'); plt.xlabel('Profit'); plt.tight_layout(); pp.savefig(fig); plt.close(fig)
    fig = plt.figure(figsize=(11.69,8.27)); plt.bar(by_region['Region'], by_region['Profit Margin']); plt.title('Profit Margin by Region'); plt.xlabel('Region'); plt.ylabel('Profit Margin'); plt.tight_layout(); pp.savefig(fig); plt.close(fig)
    fig = plt.figure(figsize=(11.69,8.27)); plt.bar(by_segment['Segment'], by_segment['Profit Margin']); plt.title('Profit Margin by Segment'); plt.xlabel('Segment'); plt.ylabel('Profit Margin'); plt.tight_layout(); pp.savefig(fig); plt.close(fig)
    fig = plt.figure(figsize=(11.69,8.27)); plt.bar(by_shipmode['Ship Mode'], by_shipmode['Profit Margin']); plt.title('Profit Margin by Ship Mode'); plt.xlabel('Ship Mode'); plt.ylabel('Profit Margin'); plt.tight_layout(); pp.savefig(fig); plt.close(fig)
    fig = plt.figure(figsize=(11.69,8.27)); plt.bar(by_disc['Discount Bucket'].astype(str), by_disc['Profit Margin']); plt.title('Profit Margin by Discount Bucket'); plt.xlabel('Discount Bucket'); plt.ylabel('Profit Margin'); plt.tight_layout(); pp.savefig(fig); plt.close(fig)
    fig = plt.figure(figsize=(11.69,8.27)); idx = np.random.choice(len(df), size=min(3000, len(df)), replace=False); plt.scatter(df['Sales'].values[idx], df['Profit'].values[idx], s=10, alpha=0.5); plt.title('Sales vs Profit (Sample)'); plt.xlabel('Sales'); plt.ylabel('Profit'); plt.tight_layout(); pp.savefig(fig); plt.close(fig)

    add_text_page('Predictive Modeling Results', [
        f"Linear Regression — R^2: {r2_lin:.3f} | RMSE: {rmse_lin:.2f} | MAE: {mae_lin:.2f}",
        f"Random Forest — R^2: {r2_rf:.3f} | RMSE: {rmse_rf:.2f} | MAE: {mae_rf:.2f}",
        'Top 10 Features:'
    ] + [f"{i+1}. {row.feature} — {row.importance:.3f}" for i, row in fi.head(10).reset_index(drop=True).iterrows()])

    pp.close()
    print('Report written to', REPORT_PATH)

if __name__ == '__main__':
    main()
