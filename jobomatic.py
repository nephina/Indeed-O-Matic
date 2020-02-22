from bs4 import BeautifulSoup as bs
import urllib.request as ur
import csv
import time
import rankingsystem as rank
import pandas
import re

SEARCH_POSITIONS = [x for x in pandas.read_csv(
    'SearchPositions.csv').to_dict()]

SEARCH_LOCATIONS = [x for x in pandas.read_csv(
    'SearchLocations.csv').to_dict()]

old_positions = read_listings()
list_of_positions = old_positions
if old_positions:
    with open('Listings.csv', mode='w') as listings:
            listing_writer = csv.writer(listings, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
            listing_writer.writerow(['Position','Company','Location','Salary',
                           'Company rating','Apply from Indeed?','Applied?',
                           'Summary','URL','SearchPosition','SearchLocation',
                           'Rating'])
    write_to_csv(list_of_positions)

elif not old_positions:
    list_of_positions = []
    print('First run of job search\n')
    with open('Listings.csv', mode='w') as listings:
            listing_writer = csv.writer(listings, delimiter=',', quotechar='"',
                                        quoting=csv.QUOTE_MINIMAL)
            listing_writer.writerow(['Position','Company','Location','Salary',
                           'Company rating','Apply from Indeed?','Applied?',
                           'Summary','URL','SearchPosition','SearchLocation',
                           'Rating'])

for position in SEARCH_POSITIONS:
    for location in SEARCH_LOCATIONS:
        scrape_jobs(position,location,list_of_positions)


rank.rank_positions()
