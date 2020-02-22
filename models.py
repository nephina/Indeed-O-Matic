from bs4 import BeautifulSoup as bs
import urllib.request as ur
import csv
import time
import rankingsystem as rank
import pandas
import re

class Listing:

    'A definition and properties of Indeed job listings.'

    EXLUDE_TERMS = [x for x in pandas.read_csv('JobExclusionTerms.csv').to_dict()]

    def __init__(self,joblisting): # Takes in single results from Indeed:
                                    # webpage.find_all(class_='result')
        self.joblisting = joblisting

    def jobtitle(self):
        link = self.joblisting.find(class_="turnstileLink")
        try:
            jobtitle = link.get('title')
        except:
            jobtitle = ''
        return jobtitle

    def company(self):
        try:
            company = self.joblisting.find(
                        class_='company').get_text().strip()
        except:
            company = ''
        return company

    def city(self):
        try:
            city = self.joblisting.find(
                        class_='location accessible-contrast-color-location'
                        ).get_text().strip()
            city = re.sub(r'[^\S]+\([^)]*\)', '', city)
        except:
            city = ''
        return city

    def salary(self):
        try:
            salary = self.joblisting.find(
                class_='salaryText').get_text().strip()
        except:
            salary = ''
        return salary

    def rating(self):
        try:
            rating = self.joblisting.find(
                class_='ratingsContent').get_text().strip()
        except:
            rating = ''
        return rating

    def can_apply_w_Indeed(self):
        try:
            if self.joblisting.find(class_='iaLabel') is not None:
                can_apply_w_Indeed = True
            else: can_apply_w_Indeed = False
        except:
            can_apply_w_Indeed = False
        return can_apply_w_Indeed

    def summary(self):
        try:
            link = self.joblisting.find(class_="turnstileLink")
            jobpage,jobhtml_doc = validate_url("http://www.indeed.com"
                                                +link.get('href'))
            summary = jobpage.find(
                class_='jobsearch-jobDescriptionText').get_text().strip()
        except:
                summary = ''
        return summary

    def url(self):
        link = self.joblisting.find(class_="turnstileLink")
        url = "http://www.indeed.com"+link.get('href')
        return url

    def position_qualifies(self):
        position = self.jobtitle().lower()
        #Define a function to check if a job title is worth checking out
        for word in self.EXLUDE_TERMS:
            if word.lower() in position: return False
        return True
    def check_if_expired(self):
        try:
            link = self.joblisting.find(class_="turnstileLink")
            jobpage,jobhtml_doc = validate_url("http://www.indeed.com"
                                                +link.get('href'))
            if jobpage.find(class_='jobsearch-JobInfoHeader-expiredHeader'):
                return False
            else:
                return True
        except:
                return True

def record_position(position,location,
foundjob,list_of_positions,current_search_positions):
    this_position = [foundjob.jobtitle(),
                     foundjob.company(),
                     foundjob.city(),
                     foundjob.url()]

    already_found = False
    for row in list_of_positions:
        if row[0:3] == this_position[0:3] or row[8] == this_position[3]:
            already_found = True
            break
    for row in current_search_positions:
        if row[0:3] == this_position[0:3] or row[8] == this_position[3]:
            already_found = True
            break
    if not already_found:
            current_search_positions.append(
                [this_position[0],
                 this_position[1],
                 this_position[2],
                 foundjob.salary(),
                 foundjob.rating(),
                 foundjob.can_apply_w_Indeed(),
                 'No',
                 foundjob.summary(),
                 this_position[3],
                 position,
                 location,
                 1])

            print(this_position[0],
                  this_position[1],
                  this_position[2])
    return current_search_positions


def search_job_page(position,location,webpage,html_doc,list_of_positions):
    current_search_positions = []
    total_results = webpage.find(id="searchCountPages").get_text()
    last_page = int(total_results
        [total_results.index("of")+2:total_results.index("jobs"
            )].strip().replace(',', ''))

    jobs_per_page = 10
    for pgno in range(0,last_page,jobs_per_page):
        if pgno > 0:
            try:
                webpage,html_doc = validate_url(
                    'https://www.indeed.com/jobs?q='+position+'&l='
                    +location+'&start='+str(pgno))
            except:
                print('Failed to read page')
        for job in webpage.find_all(class_='result'):

            foundjob = Listing(job)

            if(foundjob.position_qualifies()):
                current_search_positions = record_position(
                    position,location,foundjob,
                    list_of_positions,current_search_positions)

    return current_search_positions

def validate_url(url):
    try:
        response = ur.urlopen(url)
        html_doc = response.read()
        webpage = bs(html_doc, 'lxml')
        return webpage,html_doc

    except:
        print('\nFailed to open:\n'+url+'\n')

def write_to_csv(positions):
    try:
        with open('Listings.csv', mode='a') as listings:
            listing_writer = csv.writer(listings, delimiter=',', quotechar='"',
                                         quoting=csv.QUOTE_MINIMAL)

            for row in positions:
                listing_writer.writerow(row)
    except:
        print('Nothing to write to file')

def scrape_jobs(position,location,list_of_positions):
    webpage,html_doc = validate_url('https://www.indeed.com/jobs?q='+position
                                    +'&l='+location+'&start=0')
    current_search_positions = search_job_page(position,location,webpage,
        html_doc,list_of_positions)
    if current_search_positions:
        for row in current_search_positions:
            list_of_positions.append(row)
        print('\nwriting positions from search of \"'+position.replace(
            '+', ' ')+'\" in \"'+location.replace('+', ' ')+'\"\n')
        write_to_csv(current_search_positions)
    else:
        print('No positions to write from search of \"'+position.replace(
            '+',' ')+'\" in \"'+location.replace('+', ' ')+'\"\n')


def read_listings():
    try:
        listings = pandas.read_csv('Listings.csv').to_dict()
        columns = ['Position','Company','Location','Salary',
                   'Company rating','Apply from Indeed?','Applied?',
                   'Summary','URL','SearchPosition','SearchLocation',
                   'Rating']
        positions = []
        for row in range(len(listings['Position'])):
            if str(listings['Applied?'][row]).lower() == 'yes':
                position = []
                for column in range(len(columns)):
                    position.append(listings[columns[column]][row])
                positions.append(position)
        print('You have applied to {} positions.\n'.format(len(positions)))
        return positions
    except:
        return None
