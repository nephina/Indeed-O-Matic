import pandas
import csv

def summary_multiplier(positions):
    summary_terms = pandas.read_csv('Summary_Weighted_Terms.csv').to_dict()
    print('Applying summary key terms multiplier')
    for row in range(len(positions['Summary'])):
        for word in range(len(summary_terms['Term'])):
            if str(summary_terms['Term'][word]) in str(positions['Summary'][row]):
                positions['Rating'][row] = float(positions['Rating'][row])*float(summary_terms['Rating'][word])
    return(positions)


def company_multiplier(positions):
    company_rating = pandas.read_csv('Company_Rating_Data.csv').to_dict()
    print('Applying company multiplier')
    for row in range(len(positions['Rating'])):
        for company_listing in range(len(company_rating['Rating'])):
            if str(positions['Company'][row]) == str(company_rating['Company'][company_listing]):
                positions['Rating'][row] = float(positions['Rating'][row])*float(company_rating['Rating'][company_listing])
    return positions

def company_rating_multiplier(positions):
    print('Applying company rating multiplier')
    for row in range(len(positions['Company rating'])):
        if float(positions['Company rating'][row]) > 0.0:
            positions['Rating'][row] = float(positions['Rating'][row])*(1+(0.1*float(positions['Company rating'][row])))
        else:
            positions['Rating'][row] = float(positions['Rating'][row])*1.3
    return positions

def rank_positions():
    positions = pandas.read_csv('Listings.csv').to_dict()
    positions = company_multiplier(positions)
    positions = summary_multiplier(positions)
    positions = company_rating_multiplier(positions)
    write_to_csv(positions)

def write_to_csv(positions):
    try:
        with open('Listings.csv', mode='w') as listings:
            listing_writer = csv.writer(listings, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
            listing_writer.writerow(['Position','Company','Location','Salary','Company rating','Apply from Indeed?','Summary','URL','SearchPosition','SearchLocation','Rating'])
            for row in range(len(positions['Position'])):
                listing_writer.writerow([positions['Position'][row],
                                         positions['Company'][row],
                                         positions['Location'][row],
                                         positions['Salary'][row],
                                         positions['Company rating'][row],
                                         positions['Apply from Indeed?'][row],
                                         positions['Summary'][row],
                                         positions['URL'][row],
                                         positions['SearchPosition'][row],
                                         positions['SearchLocation'][row],
                                         positions['Rating'][row]])
    except:
        print('Nothing to write to file')
