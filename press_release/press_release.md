# Most Americans Don't Know Their Air Will Be Unhealthy Tomorrow. A New Machine Learning Tool Can Change That.

## Hook

Every day, millions of Americans — asthma sufferers, parents of young children, elderly residents — step outside without knowing whether the air they're about to breathe will harm them. By the time a "Code Orange" air quality alert reaches the evening news, sensitive individuals have often already spent hours outside. A new data science project shows that tomorrow's air quality can be predicted today, with nearly 80% accuracy, using only the data the U.S. government already collects.

## Problem Statement

Air pollution from fine particulate matter, known as PM2.5, is one of the most serious environmental health threats in the United States. These microscopic particles — small enough to lodge deep in the lungs and enter the bloodstream — are linked to asthma attacks, heart disease, and tens of thousands of premature deaths each year. The Environmental Protection Agency already operates a vast network of outdoor monitoring stations that measure PM2.5 daily, and translates those readings into the familiar color-coded Air Quality Index (AQI) that ranges from "Good" to "Hazardous."

The problem is timing. Current AQI values tell people what the air *is* doing right now, not what it *will* be doing tomorrow. For someone with a child who has asthma, or an elderly parent with heart disease, knowing today that tomorrow will be a "Code Orange" day is the difference between a safe morning walk and an emergency inhaler. Despite having thousands of monitoring sites, reliable next-day forecasts of neighborhood-level AQI are surprisingly hard to find — especially for the general public.

## Solution Description

This project builds a simple forecasting tool that learns from the EPA's own monitoring data. Using a full year of daily PM2.5 measurements from hundreds of sites across ten U.S. states, stored in a modern cloud database, a machine learning model was trained to look at the last week of readings at any given location and predict what tomorrow's AQI category will be.

The result: the model correctly predicts next-day AQI category about **four out of every five days**, with no weather data, no satellite imagery, and no expensive sensors required. The biggest surprise from the model was that **how much air quality was jumping around during the past week** turned out to be almost as important as the pollution level itself. In other words, unstable air patterns — the kind caused by shifting winds, an approaching wildfire plume, or a passing weather front — are an early warning sign of a bad air day ahead.

For communities, this means that tomorrow's air quality forecast can be produced automatically from data the government already publishes, giving families, schools, and public health officials a full day to prepare. For vulnerable individuals, that head start can mean skipping a morning jog, moving a soccer practice indoors, or making sure a rescue inhaler is within reach — simple decisions that save lives.

## Chart

![Daily PM2.5 by State — EPA AQS 2023](figures/pm25_by_state.png)

*Daily PM2.5 concentrations vary sharply across U.S. states, with Western states like Washington, Oregon, and Idaho showing the highest levels and widest fluctuations — largely driven by wildfire smoke events. The red dashed line marks the World Health Organization's recommended daily limit of 15 µg/m³; most states exceeded this threshold on a significant share of days in 2023. These state-by-state differences are exactly the patterns the forecasting model learns to capture, allowing it to tailor its predictions to the distinct air quality profile of each region.*
