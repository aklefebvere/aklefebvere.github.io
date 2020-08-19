---
layout: post
title: BudgetBlocks budgeting app
subtitle: A DS API that autocategorizes transactions and more!
img: /img/app_icon.svg
---

Over the course of 8 weeks, I worked with a cross-functional team consisting of: web developers, an iOS developer, a UI/UX designer, and another data scientist to create a budgeting app called BudgetBlocks.
As the data science team, we built and deployed (AWS EB) an API, using FastAPI, that automatically categories transactions and more.

Interact with the API directly [here](https://api.budgetblocks.org/docs).

## Here is a high level break down of what the API can do and how it works:
* ### Recategorization of transactions  
Allows users to quickly and easily get their transactions grouped into categories, also can tweak how transactions are categorized if they desire.
  1. Users link their bank accounts with the Plaid financial service.
  2. Plaid uses a ML model to tag transactions with one of 350+ categories.
  3. The API recategorizes these transactions into the 9 default BudgetBlocks categories or a user's customized categories.
* ### Census comparison  
Allow users to quickly see how their spending and budget goals stack up against the average of their nearest city.
  1. On the front end, users input their state and city.
  2. Uses GeoPy library to convert the location to coordinates.
  3. Implements [the Haversine formula](https://en.wikipedia.org/wiki/Haversine_formula) to find the closest city that there is Census data for.
  4. Returns 2017-18 Census average expenditure data for that city, categorized according to a user's preferences.
* ### Admin GUI  
As this project may not have continued DS support, we built this admin interface to allow a future development team to edit how transactions are categorized by default.
  1. Login [here](https://api.budgetblocks.org/admin).
  2. At this interface they can:
  
    - Adjust how transactions are recategorized from Plaid categories to BudgetBlocks categories  
    - View the current status of the PostgreSQL database that's used to make the recategorizations  
    - Reset the default recategorization database to its original status
    - Reset the database holding users' customized preferences
    - View a timestamped changelog of changes to default categorization prefences  

The GitHub repository for this project can be found [here](https://github.com/Lambda-School-Labs/budget-blocks-ds). The reademe file includes multiple GIFs showing the API routes and Admin Interface in action.
