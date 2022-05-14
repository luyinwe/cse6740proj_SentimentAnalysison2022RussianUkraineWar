# Introduction

Starting from Feb 24th, 2022, Russia began an open-military conflict of Ukraine. From then on, related rumors spread online and make public confused about what is actually going on in the frontline. In this project we tend to figure out what people on social media care about, and their opinions about the conflict.



# Methodology

We plan to prepare a new dataset relating to this issue from twitter and label the dataset automatically with different sentimental analysis tools. Then analyze the relationship between the results and data characteristics like date, location, keyword and sentiment. Finally we will apply Latent Dirichlet Allocation(LDA) model to different groups and develop a UI to visualize the result.



## LDA Visualization

 ![](image\lda_result.png)

LDA models can help us extract people's opinions on specific issues. For example, if we select the names of some leaders (Biden, Putin, Zelensky) as keywords, we can get a rough idea of what people think of them. Take Biden as an example, people mentioned more about his altitude towards Russia and China. Some of them also asked Biden to prevent Russia from invading Ukraine. Quite a few people compared Biden with Trump.

## Visualization Panel

![](image\visualization_panel.png)

The UI system is designed to visualize how public views variate across the nations, and how it changes as the war still going on, and understand what people really care about. The bluer the coutry is, the more positive the public view is in this country. 

For example, if we set the date to Feb 10th, as we can see from Figure 6, three keywords about the news, “belarus”,“Ukraine” and “Russia” are catched by our LDA model with neutral corpus. The extracted keywords match the news on Feb 10th that Russia starts a maneuver with Belarus.



# Important Reference

1. Dataset

https://www.kaggle.com/datasets/foklacu/ukraine-war-tweets-dataset-65-days?resource=download

2. Sentimental Analysis tool

https://github.com/pysentimiento/pysentimiento

3. LDA analysis

https://arxiv.org/abs/2109.12372

Know it to Defeat it: Exploring Health Rumor Characteristics and Debunking Efforts on Chinese Social Media during COVID-19 Crisis