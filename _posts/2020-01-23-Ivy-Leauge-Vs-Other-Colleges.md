---
layout: post
title: Ivy Leauge vs. Other Colleges
image: /img/ivy-league-emblems.jpg
---
# Introduction
One of the first things high school students think about is what college to attend after high school. Some of the questions high school students ask are: 
  * Should I strive for excellent grades and perfect ACT/SAT scores to get into an Ivy Leauge school? 
  * Should I get pretty good grades and just go for a college in my state? 
  * Is there really a big difference in students salaries that graduate from a Ivy Leauge school compared to any other school? 
  
This blog post answers these questions high school students have thought about. I will be analyzing and visualizing this [dataset](https://www.kaggle.com/wsj/college-salaries#salaries-by-college-type.csv) to compare the Ivy League salaries throughout a graduates career compared to other colleges in the United States. This blog does not take in account what career pathway a student would want to pursue, this blog is solely looking at just the money aspect of colleges.

![](https://i.gyazo.com/0a30944743f0f6dbddb0d32c42be1229.png)

The image above shows the raw dataset I will be working with. This dataset consists of 269 schools all across the United States. Each school is categorized into five categories: Ivy Leauge, engineering, liberal arts, party (I'm not really sure how a school is considered a "party college"), and state. Each school contains the average mean salary throughout a graduates career (starting salary to mid-career 90 percentile). 

# Ivy League Salaries vs other colleges Salaries

The first thing I want to do is to compare the mean average salary for each salary category from each school type. I split the dataset into five different datasets that only have the data for each category of schools so I could find the mean of each salary category. I then created a grouped bar chart to visualize the mean for each salary category from each school type.

![](/img/Placeholder_mean_graph.png)

The x-axis is each salary category and the percentages are the percentile of mid-career salaries (ex: 10% means the salary mean is higher than 10% of mid-career salaries but lower than 90% of salaries). By looking at the graph, you can tell that Ivy League schools have the highest mean salary for 5/6 of the salary categories. Engineering colleges are very close to Ivy League colleges from starting mean salary to mid-career 25 percentile then Ivy League schools shoot up in salary after the 25 percentile.

We can analyze how close the means between Ivy League and Engineering are by plotting just the two categories in their own graphs.

![](/img/Placeholder_ivy_eng_starting_Salary.png)

![](/img/Placeholder_ivy_eng_midcareer25_Salary.png)

With the graphs above, we can further prove that the Ivy League mean salaries is slightly higher by an insignificant amount compared to engineering colleges.

# Ivy Leauge Salaries vs Engineering and Liberal Arts Salaries combined

To further prove the significance of difference between Ivy Leauge Schools and the other schools in the United States, we can get the second and third highest mean salaries in the data and get the mean of the two and compare it to the Ivy League Salaries by plotting it on a graph.


