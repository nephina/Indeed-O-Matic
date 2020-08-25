# Readme

## How To Use

### Setup

First make sure you have all the relevant packages:

```
$ pip install bs4 urllib csv pandas re torch torchtext nltk scikit-learn numpy PyQt5
```

{% hint style="info" %}
 You may need to use pip3 if pip defaults to Python 2, this code is written in Python 3.6 and has not been tested in any other version
{% endhint %}

To start, you will need to set up your SearchLocations and SearchPositions files. Simply enter the cities you want to search, and some generic job types that you are looking for.

Next, run the code for the first time:

```text
$ python jobomatic.py
```

{% hint style="info" %}
You may need to use the command _python3_ if your system defaults to python2. Check which version it defaults to with _python --version_
{% endhint %}

Depending on how many cities and search positions you entered, this could take minutes to hours. Once it is done scraping Indeed for jobs, it will output a new Listings file, and then open up a window where it will prompt you to select one of the jobs.

### Personalization

The job listings it scraped from Indeed come in no particular order, so they are not ranked yet. That is where you come in. The window will prompt you to select your favorite of two jobs. If you find yourself spending a lot of time choosing between the two, or they are both jobs you don't like, use the "skip" button. In the backend there is an AI, specifically a neural network, which will be learning how to rank the jobs based on your preferences. If you are having trouble telling the difference, that's a good sign that it's getting it right and doesn't need to learn anything else about those jobs. And if both jobs are bad, there is no need to fill the training data with information about them because you won't care how they get ranked and they will end up at the bottom anyway.  
  
The AI takes input and trains in sequential steps. Each step takes 50 pairwise selections from you, and then takes some time to train the AI. At each step, you should notice the pairs of jobs becoming both better-suited to your tastes and harder to choose between. You will also notice that the AI takes longer to train as its training data builds and as it also grows in size to accommodate the new information.

At each step, you can open the Listings.csv file to check how well it is doing at ranking the jobs. I have found that after a couple of steps, the AI has a very rough grasp of what you want. At 4 steps, the AI is good, but not great. At around 7 or 8 steps, you will start noticing that it puts jobs up top which at first you might think are errors, but if you take a closer look at the job description you are more intrigued. I haven't tested it past 8 steps, but I'm not sure there's much benefit to going past that, but feel free to explore.  
  
Note that this can take a while, depending on whether you're a fast or slow reader, or whether you're detail-oriented or a text-skimmer. To get through 8 steps of training, you should probably set aside an evening. The good thing is that after that training is done, you never need to train it again, unless your tastes change in a significant way and you want to retrain it. You can scrape any number of new jobs and the AI you trained once will always be able to rank them correctly.  
  
Have fun, happy job searching!

