# AI-Powered Multi-Modal Dengue Outbreak Prediction System

This project implements an AI-powered multi-modal ensemble model to predict dengue outbreak risks with a 7-day early warning window. The system integrates heterogeneous data sources to identify potential outbreak hotspots and enable proactive public health responses.

# Key Features

Multi-modal data integration including weather patterns, population density, human mobility trends, social media signals, and healthcare data

Weekly temporal aggregation and feature engineering

Ensemble-based machine learning pipeline using XGBoost

Zone-wise dengue risk scoring with hotspot prediction

7-day advance outbreak early warning system

# Geographic Coverage

The current implementation covers five major cities in Tamil Nadu, India:

Chennai

Coimbatore

Kanyakumari

Salem

Tiruchirapalli

The architecture is scalable and can be extended to additional cities or regions.

# Data Sources

India Meteorological Department (IMD) weather datasets

Census of India population and demographic data

Historical dengue incidence and healthcare records

Open mobility datasets and social media-derived signals

Machine Learning Pipeline

Data ingestion and preprocessing from multiple open sources

Feature extraction and weekly aggregation

Model training and inference using XGBoost

Generation of zone-wise dengue risk scores

# Outputs and Visualization

Weekly dengue outbreak risk maps

Zone-level risk scores for each city

Interactive spatial visualization using Mapbox

Technology Stack

Programming Language: Python

Machine Learning: XGBoost

Data Processing: Pandas, NumPy

Visualization: Mapbox
