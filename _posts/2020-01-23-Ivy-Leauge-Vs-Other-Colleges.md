---
layout: post
title: Ivy League vs. Other Colleges
image: /img/ivy-league-emblems.jpg
---
# Introduction
One of the first things high school students think about is what college to attend after high school. Some of the questions high school students ask are: 
  * Are my grades good enough to get into an Ivy League school?
  * Is it even worth going to an Ivy League School?
  * Should I get decent enough grades and just go for a good college in my state? 
  * Is there really a big difference in students salaries that graduate from a Ivy League school compared to any other school? 
  
This blog post answers these questions high school students have thought about. I will be analyzing and visualizing this [dataset](https://www.kaggle.com/wsj/college-salaries#salaries-by-college-type.csv) to compare the Ivy League salaries throughout a graduates career compared to other colleges in the United States. Before analysing the data, I believe that there will not be a significant difference in Ivy League salaries compared to other colleges in the United States. This blog does not take in account what career pathway a student would want to pursue. This blog only looks at the average monetary earning from individuals that graduated from a college.

![](https://i.gyazo.com/0a30944743f0f6dbddb0d32c42be1229.png)

The image above shows the raw dataset I will be working with. This dataset consists of 269 schools all across the United States. Each school is categorized into five categories: Ivy League, engineering, liberal arts, party (I'm not really sure how a school is considered a "party college"), and state. Each school contains the average mean salary throughout a graduates career (starting salary to mid-career 90 percentile). 

# Ivy League Salaries vs. other colleges Salaries

The first thing I want to do is to compare the mean average salary for each salary category from each school type. I split the dataset into five different subsets that only have the data for each category of schools so I could find the mean of each salary category. I then created a grouped bar chart to visualize the mean for each salary category from each school type.

![](/img/FINAL_mean_graph.png)

The x-axis is each salary category and the percentages are the percentile of mid-career salaries (ex: 10% means the salary mean is higher than 10% of mid-career salaries but lower than 90% of salaries). By looking at the graph, you can tell that Ivy League schools have the highest mean salary for five out of six of the salary categories. Engineering colleges are very close to Ivy League colleges from starting mean salary to mid-career 25 percentile then Ivy League schools shoot up in salary afterwards.

We can analyze how close the means between Ivy League and Engineering are by plotting the two categories in their own graphs.

![](/img/FINAL_ivy_eng_starting_salary.png)

![](/img/FINAL_ivy_eng_midcareer25_Salary.png)

With the graphs above, we can further prove that the Ivy League mean salaries is slightly higher by an insignificant amount compared to engineering colleges.

# Ivy League Salaries vs. Engineering and Liberal Arts Salaries Combined

To further prove the significance of difference between Ivy League Schools and the other schools in the United States. To do this, we can get the second and third highest mean salaries in the data and get the mean of the two and compare it to the Ivy League Salaries. Once we get the means, we can plot it on a graph.

![](/img/FINAL_ivy_eng_arts_mean_graph.png)

Even with combining two high mean salaries, Ivy League schools still come out on top. If we calculate the pecentage increase for the 75th percentile and 90% percentile of salaries, we can see how significant the difference are. Ivy League 75th percentile has a 27% increase in salary and Ivy League 90th percentile has a 19% increase in salary when compared to Engineering colleges and Liberal Arts Colleges salaries combined.

# Conclusion

My hypothesis have been proven incorrect by the data we analyzed. By just looking at the data, there is a significant difference in Ivy League salaries compared to other colleges in the United States. The outcome of this data could change if we take account of what career pathway a graduate wants to pursue. In conclusion, if you go to an Ivy League school and you work very hard to get lucky enough to get a very well paying job after you graduate, it is worth attending an Ivy League school.

[Colab Link](https://colab.research.google.com/drive/1F3poZJOL9j2h3coI-7PtGFpOBYosuOn7)
