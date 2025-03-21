# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 00:54:00 2025

@author: Lenovo
"""


import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import metrics
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import plotly.graph_objects as go

# Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Load data
forecast_df = pd.read_csv('forecast_data.csv')
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

real_df = pd.read_csv('real_results.csv')
real_df.rename(columns={'Power_kW': 'Power (kW) [Y]'}, inplace=True)
real_df['Date'] = pd.to_datetime(real_df['Date'])

merged_df = pd.merge(forecast_df, real_df[['Date', 'Power (kW) [Y]']], on='Date', how='left')

unique_columns = sorted(set([col for col in merged_df.columns if "date" not in col.lower()]))
# Load models
with open('NN_model.pkl', 'rb') as file:
    nn_model = pickle.load(file)
with open('RF_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Feature extraction
features = forecast_df.iloc[:, 1:15]
X_values = features.values
y_real = real_df['Power (kW) [Y]'].values

# Model predictions
y_pred_nn = nn_model.predict(X_values)
y_pred_rf = rf_model.predict(X_values)

# Error metrics calculation
def compute_metrics(y_true, y_pred):
    return {
        "MAE": metrics.mean_absolute_error(y_true, y_pred),
        "MBE": np.mean(y_true - y_pred),
        "MSE": metrics.mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        "cvRMSE": np.sqrt(metrics.mean_squared_error(y_true, y_pred)) / np.mean(y_true),
        "NMBE": np.mean(y_true - y_pred) / np.mean(y_true)
    }

nn_metrics = compute_metrics(y_real, y_pred_nn)
rf_metrics = compute_metrics(y_real, y_pred_rf)

metrics_df = pd.DataFrame({
    "Methods": ["Neural Network", "Random Forest"],
    **{key: [nn_metrics[key], rf_metrics[key]] for key in nn_metrics}
})

#Define the feature selection method
def feature_selection(k, score_func, selected_features):
    features = SelectKBest(k=k, score_func=score_func)
    fit = features.fit(forecast_df[selected_features], y_real)
    return fit.scores_, selected_features


forecast_results = real_df[['Date', 'Power (kW) [Y]']].copy()
forecast_results["Neural Network"] = y_pred_nn
forecast_results["Random Forest"] = y_pred_rf

# Precompute Figures
initial_features_fig = px.line(forecast_df, x='Date', y=forecast_df.columns[1:15])
initial_forecast_fig = px.line(forecast_results, x='Date', y=["Power (kW) [Y]", "Neural Network", "Random Forest"])

# Initialize app with suppress_callback_exceptions
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server
app.layout = html.Div([
    html.H1("IST Energy Forecast Tool"),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Features', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Error Metrics', value='tab-3'),
        dcc.Tab(label='Scatter Plot', value='tab-4'),
        dcc.Tab(label='Feature Selection Method', value='tab-5')
    ]),
    html.Div(id='date-picker-container'),
    dcc.Store(id='features-fig-store', data=initial_features_fig.to_dict()),
    dcc.Store(id='forecast-fig-store', data=initial_forecast_fig.to_dict()),
    html.Div(id='tabs-content')
])

@app.callback(Output('date-picker-container', 'children'),
              Input('tabs', 'value'))
def update_date_picker(tab):
    if tab in ['tab-1', 'tab-2']:
        return dcc.DatePickerRange(
            id='date-picker',
            start_date=forecast_results['Date'].min(),
            end_date=forecast_results['Date'].max(),
            display_format='YYYY-MM-DD'
        )
    return None

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.Label("Select Features to Display:"),  
            dcc.Dropdown(
                id='features-dropdown-tab1',  # Unique ID for Tab 1
                options=[{'label': col, 'value': col} for col in forecast_df.columns[1:]],  # Use relevant columns
                value=[forecast_df.columns[1], forecast_df.columns[2]],  # Default selection
                multi=True  # Allow multiple selections
            ),
            dcc.Graph(id='features-graph')
        ])

    elif tab == 'tab-2':
        return html.Div([
            html.Label("Select Forecast Models: "),
            dcc.Dropdown(
                id='forecast-dropdown',
                options=[
                    {'label': 'Raw Power', 'value': 'Power (kW) [Y]'},
                    {'label': 'Neural Network', 'value': 'Neural Network'},
                    {'label': 'Random Forest', 'value': 'Random Forest'}
                ],
                value=['Power (kW) [Y]', 'Neural Network', 'Random Forest'],
                multi=True
            ),
            dcc.Graph(id='forecast-graph', figure=initial_forecast_fig)
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.Label("Select Metrics: "),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[{'label': metric, 'value': metric} for metric in metrics_df.columns[1:]],
                value=metrics_df.columns[1:].tolist(),
                multi=True
            ),
            html.Table(id='metrics-table')
        ])
    elif tab == 'tab-4':
        unique_columns = sorted(set(
            [col for col in merged_df.columns if "date" not in col.lower()]
        ))
        return html.Div([
            html.H3("Custom Scatter Plot"),
            
            html.Label("Select X-axis:"),
            dcc.Dropdown(
                id='x-axis',
                options=[{'label': col, 'value': col} for col in unique_columns],
                value="Power (kW) [Y]"
            ),
            
            html.Label("Select Y-axis:"),
            dcc.Dropdown(
                id='y-axis',
                options=[{'label': col, 'value': col} for col in unique_columns],
                value="Power-1"
            ),
            
            #html.Button("Plot Graph", id='plot-button', n_clicks=0),
            
            dcc.Graph(id='scatter-plot')
        ])
    elif tab == 'tab-5':
        return html.Div([
            html.Label("Select Features:"),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in forecast_df.columns[1:]],
                multi=True,
                value=forecast_df.columns[1:9]
            ),
            html.Label("Select Feature Selection Method:"),
            dcc.RadioItems(
                id='score-function',
                options=[
                    {'label': 'F Regression', 'value': 'f_regression'},
                    {'label': 'Mutual Information Regression', 'value': 'mutual_info_regression'}
                ],
                value='mutual_info_regression'
            ),
            html.Label("Select K Best Features:"),
            dcc.Input(id='k-value', type='number', value=3, min=1, max=len(forecast_df.columns)-1),
            html.Button("Apply", id='apply-button', n_clicks=0),
            dcc.Graph(id='feature-selection-graph')
        ])



@app.callback(
    Output('features-graph', 'figure'),
    [Input('features-dropdown-tab1', 'value'),
     Input('date-picker', 'start_date'),
     Input('date-picker', 'end_date')]
)
def update_features_graph(selected_features, start_date, end_date):
    if not selected_features:  # Ensure at least one feature is selected
        return px.line(title="Please select at least one feature.")

    # Filter forecast_df based on selected date range
    filtered_df = forecast_df[(forecast_df['Date'] >= start_date) & (forecast_df['Date'] <= end_date)]

    # Create the line chart with the selected features
    fig = px.line(filtered_df, x='Date', y=selected_features, title="Selected Features Over Time")
    return fig


@app.callback(Output('forecast-graph', 'figure'),
              [Input('forecast-dropdown', 'value'), Input('date-picker', 'start_date'), Input('date-picker', 'end_date')])
def update_forecast(selected_models, start_date, end_date):
    fig = px.line(forecast_results, x='Date', y=selected_models)
    fig.update_xaxes(range=[start_date, end_date])
    return fig

@app.callback(Output('metrics-table', 'children'),
              [Input('metric-dropdown', 'value')])
def update_metrics_table(selected_metrics):
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in ['Methods'] + selected_metrics])),
        html.Tbody([
            html.Tr([html.Td(metrics_df.iloc[i][col]) for col in ['Methods'] + selected_metrics]) for i in range(len(metrics_df))
        ])
    ])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis', 'value'), Input('y-axis', 'value')]
)
def update_graph(x_col, y_col):
    if not x_col or not y_col:
        return px.scatter(title="Please select valid columns")
    
    combined_df = merged_df[[x_col, y_col]].dropna()
    
    fig = px.scatter(combined_df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
    return fig

@app.callback(
    Output('feature-selection-graph', 'figure'),
    [Input('apply-button', 'n_clicks')],
    [dash.dependencies.State('feature-dropdown', 'value'),
     dash.dependencies.State('score-function', 'value'),
     dash.dependencies.State('k-value', 'value')]
)
def update_feature_selection(n_clicks, selected_features, score_func, k):
    if score_func == 'f_regression':
        score_function = f_regression
    else:
        score_function = mutual_info_regression
    
    scores, features = feature_selection(k, score_function, selected_features)
    
    fig = go.Figure([go.Bar(x=features, y=scores)])
    fig.update_layout(title="Feature Scores", xaxis_title="Features", yaxis_title="Score")
    return fig

if __name__ == '__main__':
    app.run_server()
