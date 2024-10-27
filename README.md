# PRC Data Challenge 2024 - Eurocontrol

## Project Overview

This repository contains a collection of scripts developed for the **PRC Data Challenge 2024** hosted by **Eurocontrol**. The goal of this project is to create a robust model for estimating the **take-off weight (TOW)** of airplanes. Using a foundational dataset, this project appends critical trajectory-derived variables to support precise TOW estimations.

## Dataset Creation

The dataset creation process involves calculating and appending additional metrics to flight trajectory data. Key tools and libraries used include:

- **Python 3.10**
- **Pandas** and **Pyarrow** for efficient data manipulation and processing.

The main data sources include Parquet files with flight trajectory data and CSV files containing metadata for each flight. These inputs are processed to generate various metrics essential for TOW estimation, such as:

- **Lift-off Calculations**: Time to lift-off, ground speed at lift-off, and ground speed changes.
- **Jet Stream Coefficients**: Analysis of polar and subtropical jet streams based on trajectory data.
- **Weather Data**: Extraction of temperature, humidity, wind components, and altitude for modeling environmental effects on flight dynamics.

Dataset creation was conducted by [**Andrey Belkin**](https://www.linkedin.com/in/andrey-belkin-03ab241ba) on a local machine with the following specifications:

- **Processor**: AMD Ryzen 7 3700X
- **GPU**: Nvidia RTX 2060

## Statistical Analysis

The statistical analysis, which examines the correlations and patterns within the data for TOW estimation, was carried out using **R** on high-performance server clusters, primarily leveraging an **NVIDIA A100-SXM4-80GB** for accelerated computation.

This analysis was led by [**Anton Bogachev**](https://www.linkedin.com/in/anton-bogachev-627353182/), focusing on deriving actionable insights and refining the dataset for model training.

## System Requirements

To replicate or extend the analysis, the following environment and hardware are recommended:

- **Python**: Version 3.10
- **Libraries**: Pandas, Pyarrow
- **Software**: R 4.4.2 for statistical analysis
- **Hardware**:
    - Local machine: AMD Ryzen 7 3700X, Nvidia RTX 2060
    - Server cluster (for large-scale computations): NVIDIA A100-SXM4-80GB

## File Descriptions

- **`construct_dataset_v4.py`**: The main controller script for data processing, this file manages dataset selection and error logging.
- **`read_parquet_v3.py`**: This script handles reading and filtering Parquet files, preparing the data for lift-off and jet stream calculations.
- **`total_g_v3.py`**: Computes flight metrics, including ground speed and weather influences, which are essential for TOW estimation.

## Contributors

- [**Andrey Belkin**](https://www.linkedin.com/in/andrey-belkin-03ab241ba): Led the dataset creation, coding, and initial variable extraction process.
- [**Anton Bogachev**](https://www.linkedin.com/in/anton-bogachev-627353182/): Conducted statistical analysis, fine-tuning the dataset for effective TOW modeling.
