import requests
from bs4 import BeautifulSoup
import re
import csv


def scrape_for_scraps(site, html_regex_filter, tag=None, iterate_trimmed=0, nested_string_tag=None,
                      csv_header=None, output_csv=None):

    lst = []

    # use requests get function to obtain the article's webpage
    request = requests.get(site)

    # convert article webpage object to html
    data = request.text

    # convert the html to a BeautifulSoup object (nested data structure)
    soup = BeautifulSoup(data, 'html.parser')

    # trim html to only display relevant information
    regex = re.compile(html_regex_filter)

    if tag:
        trimmed_html = soup.body.find_all(tag, class_=regex)
    else:
        trimmed_html = soup.body.find_all(class_=regex)

    # View HTML code of article webpage in BeautifulSoup format
    if iterate_trimmed and nested_string_tag:
        for ele in trimmed_html:
            ele_list = list(map(lambda x: x.string, ele.find_all(nested_string_tag)))
            lst = lst + ele_list

    print(lst)

    if output_csv:
        lst = list(map(lambda x: [x], lst))
        with open(output_csv+'.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(csv_header)
            writer.writerows(lst)


if __name__ == "__main__":
    scrape_for_scraps(site='http://dogtime.com/dog-breeds/profiles',
                      html_regex_filter=r"^article-crumbs clearfix group-letter letter.*", iterate_trimmed=1,
                      nested_string_tag='h2',
                      csv_header=['doggo_breed'], output_csv='doggotime_breeds')