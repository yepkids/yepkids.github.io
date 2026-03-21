---
title: "Machine learning-based prediction of heat index in selected U.S. cities"
collection: publications
category: manuscripts
permalink: /publications/2026-03-01-ML_heatwave
excerpt: "We apply Random Forest and Gated Recurrent Unit models to predict next day's heat index in 4 selected U.S. cities"
date: 2026-02-01
venue: '[In preprint]. ArXiv'
#slidesurl: 'https://academicpages.github.io/files/slides3.pdf'
paperurl: 'https://yepkids.github.io/files/ML_heat_index.pdf'
#citation: 'Your Name, You. (2015). &quot;Paper Title Number 3.&quot; <i>Journal 1</i>. 1(3).'
---

Abstract
========

Heat stress has harmful effects that impact communities across the Unitedt States,
particularly when high temperatures are accompanied by high humidity. The
combined impact of temperature and humidity can be summarized by the heat
index (HI). Current state-of-the-art numerical weather prediction models are often
biased when forecasting temperature and humidity even within a 24-hour forecast
lead time. This study explores the ability of machine learning (ML) models to
accurately predict the next-day heat index using Random Forest and single-layer
Gated Recurrent Unit (GRU) models in four locations across the United States.
We find that Random Forest and GRU models perform reasonably well at all four
selected locations. Mean absolute HI error ranges from 4.5 to 6.6◦F. All model
versions have an accuracy rate exceeding 80% in three of the four locations in terms
of successfully forecasting an extreme heat day, as indicated by a high afternoon
HI. The GRU model achieves over 95% accuracy in these three locations. Model
performance details vary by location. In Minneapolis and Portland, which have
relatively few days with high HI values, models’ accuracy is high, but the recall
and precision are generally very low. In contrast, Dallas, a location with many
high HI days, shows moderately high accuracy, as well as extremely high recall
and precision. These differences are likely due to distinct causes of heatwaves
in different climatological regions of the United States, as reflected in the feature
importance scores output by Random Forest models. The ML models designed in
this study can be used to assist with local heat index forecasting and extreme heat
warning issuance at minimal computational cost

[Prepint Link](https://yepkids.github.io/files/ML_heat_index.pdf)