import pandas
import csv

def summary_multiplier(positions):
    summary_terms = pandas.read_csv('JobDescriptionTerms.csv').to_dict()
    print('Applying summary key terms multiplier')
    for row in range(len(positions['Summary'])):
        for word in range(len(summary_terms['Term'])):
            if (str(summary_terms['Term'][word])).lower() in (str(positions['Summary'][row])).lower():
                positions['Rating'][row] = float(positions['Rating'][row])*float(summary_terms['Rating'][word])
    return(positions)


def company_multiplier(positions):
    company_rating = pandas.read_csv('CompanyRatings.csv').to_dict()
    new_companies = []
    print('Applying company multiplier')
    for row in range(len(positions['Rating'])):
        found_company = False
        for company_listing in range(len(company_rating['Rating'])):
            if str(positions['Company'][row]).lower() == str(company_rating['Company'][company_listing]).lower():
                if float(company_rating['Rating'][company_listing]) >= 1:
                    positions['Rating'][row] = float(positions['Rating'][row])*float(company_rating['Rating'][company_listing])
                else:
                    positions['Rating'][row] = float(positions['Rating'][row])
                found_company = True

        if not found_company:
            if str(positions['Company'][row]).lower() not in [x.lower() for x in new_companies]:
                new_companies.append(str(positions['Company'][row]))

    if new_companies:
        with open('CompanyRatings.csv',mode='a') as companies:
                    company_writer = csv.writer(companies, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
                    for row in new_companies:
                        company_writer.writerow([row,''])
                    print('Wrote new companies to file')
    else:
        print('No new companies found')
    return positions

def company_rating_multiplier(positions):
    print('Applying company rating multiplier')
    for row in range(len(positions['Company rating'])):
        if float(positions['Company rating'][row]) > 0.0:
            positions['Rating'][row] = float(positions['Rating'][row])*(1+(0.1*float(positions['Company rating'][row])))
        else:
            positions['Rating'][row] = float(positions['Rating'][row])*1.3
    return positions

def location_rating_multiplier(positions):
    print('Applying location rating multiplier')
    for row in range(len(positions['Rating'])):
        if str(positions['SearchLocation'][row]) == 'Milwaukee':
            positions['Rating'][row] = 10*float(positions['Rating'][row])
        elif str(positions['SearchLocation'][row]) == 'Chicago':
            positions['Rating'][row] = 5*float(positions['Rating'][row])
    return positions

def rank_positions():
    positions = pandas.read_csv('Listings.csv').to_dict()
    for row in range(len(positions['Rating'])):
        positions['Rating'][row] = 1.0
    positions = company_multiplier(positions)
    positions = summary_multiplier(positions)
    positions = company_rating_multiplier(positions)
    positions = location_rating_multiplier(positions)
    write_to_csv(positions)

def write_to_csv(positions):
    try:
        with open('Listings.csv', mode='w') as listings:
            listing_writer = csv.writer(listings, delimiter=',', quotechar='"',quoting=csv.QUOTE_MINIMAL)
            listing_writer.writerow(['Position','Company','Location','Salary','Company rating','Apply from Indeed?','Applied?','Summary','URL','SearchPosition','SearchLocation','Rating'])
            for row in range(len(positions['Position'])):
                listing_writer.writerow([positions['Position'][row],
                                         positions['Company'][row],
                                         positions['Location'][row],
                                         positions['Salary'][row],
                                         positions['Company rating'][row],
                                         positions['Apply from Indeed?'][row],
                                         positions['Applied?'][row],
                                         positions['Summary'][row],
                                         positions['URL'][row],
                                         positions['SearchPosition'][row],
                                         positions['SearchLocation'][row],
                                         positions['Rating'][row]])
    except:
        print('Nothing to write to file')
