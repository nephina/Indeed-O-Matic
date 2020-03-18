# Readme

## How To Use

First make sure you have all the relevant packages:

```
$ pip install bs4 urllib csv pandas re
```

{% hint style="info" %}
 You may need to use pip3 if pip defaults to Python 2, this code is written in Python 3.6 and has not been tested in any other version
{% endhint %}

To start, you will need to set up your SearchLocations and SearchPositions files. Simply enter the cities you want to search, and some generic job types that you are looking for.

Next, run the code for the first time:

```text
$ python jobomatic.py
```

Depending on how many cities and search positions you entered, this could take minutes to hours. I suggest starting with a limited set. After this is finished running, it will output a new Listings file. This will contain all the jobs that it found. It will also have written a list of all the companies it found to CompanyRatings. 

Now that you have these job listings, it is time to start filling out your JobExclusionTerms and JobDescriptionTerms files:

1. The first is responsible for throwing out any jobs with certain terms in their job title. Your first search will probably have turned up some wildly different jobs than what you were looking for, this is where you can make sure they don't show up in future searches. Be careful with this, and use spaces if necessary because you might accidentally match something you didn't mean to \(i.e. "SR" will match other words like "grassroots" or "disrupt"\). 
2. The second contains terms that might appear in the job description and associated multipliers that are used to determine how good of a fit this job is for you. A term that you would be happy to see in the description would be assigned a number greater than 1, a term you don't want to see is assigned a number less than 1. It's up to you to determine your own range of good to bad, the ranking system just finds whether any term occurs in the job description, and multiplies the ranking by the associated multiplier.

{% hint style="info" %}
The term multipliers for any job listing have a multiplicative effect on each other, but multiple instances of the same term are not counted.
{% endhint %}

  
The CompanyRatings file is also used to rank the job listings. I use a ranking scale of 1 to 3, but you are free to use whatever scale and resolution you would like. A 1-4 or even 1-5 scale might serve you better. How it works is that every time you run the code, it will add to the file any new companies it found. It is up to you to go through it and rank them, if you don't, it will default to listing them all as "1's." It's a lot of work \(I'm up to 8k companies already!\), so it might be simplest to just ctrl-f for the companies that you already know you like, and give them high rankings, and disregard the rest. I will say that after googling thousands of companies I'd never heard of, I have a greater appreciation for both the diversity and the blandness of the US corporate world.

After you have these files filled out, it's just a matter of running it and changing it until the ranked list it spits out makes sense to you. What I like doing after running it, is that I will look at the top of the ranked list to see if there are any positions that I clearly dislike, and either enter the job title term into the JobExclusionTerms file or enter some new terms that I hadn't thought of into the JobDescriptionTerms with a low weight. Then I go to the bottom of the list, and do the same again except in reverse. Because it can take quite a long time for the scraper to run through all the cities and positions, I recommend you start with only a couple cities and positions. That way you will get a good turnaround time on your edits, and be able to arrive at a ranking system that works for you faster than otherwise.

Have fun, happy job searching!

