#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 17:46:59 2018

@author: tgadfort
"""

def generate_lift_chart(y_actual, y_predicted, n_bins):
    noise = np.random.normal(scale=1.0E-8, size=y_predicted.shape)
    df = pd.DataFrame({'y_actual': y_actual, 'y_predicted': y_predicted + noise})
    
    n_events = df['y_actual'].sum()
    actual_event_rate = df['y_actual'].mean()
    
    df['bin'] = pd.qcut(-df['y_predicted'], n_bins, labels=list(range(1, n_bins + 1)))
    grouped = df.groupby('bin')
    
    df_summary = grouped.agg({'y_actual': ['count', 'sum'], 'y_predicted':['mean']})
    df_summary.columns = ['.'.join(col) for col in df_summary.columns.values]
    df_summary.rename(columns={'y_actual.count': 'n_obs', 'y_actual.sum': 'n_events', 'y_predicted.mean': 'avg_score'}, inplace=True)
    
    df_summary['captured_rate'] = df_summary['n_events'] / n_events
    df_summary['cumulative_captured_rate'] = df_summary['captured_rate'].cumsum()
    df_summary['actual_event_rate'] = df_summary['n_events'] / df_summary['n_obs']
    df_summary['cumulative_actual_event_rate'] = df_summary['n_events'].cumsum() / df_summary['n_obs'].cumsum()
    df_summary['cumulative_avg_score'] = (df_summary['avg_score'] * df_summary['n_obs']).cumsum() / df_summary['n_obs'].cumsum()
    df_summary['lift'] = df_summary['actual_event_rate'] / actual_event_rate
    df_summary['cumulative_lift'] = df_summary['cumulative_actual_event_rate'] / actual_event_rate

    df_summary = df_summary[['n_obs', 'n_events', 'captured_rate', 'cumulative_captured_rate', 'avg_score', 'actual_event_rate', 'cumulative_avg_score', 'cumulative_actual_event_rate', 'lift', 'cumulative_lift']]

    df_summary.reset_index(inplace=True)
    
    return df_summary
