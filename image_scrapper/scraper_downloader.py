import requests
from bs4 import BeautifulSoup
import re
import csv
from googleapiclient.discovery import build
from pprint import pprint
import time
import random


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


def photo_downloader(url, tag, folder):
    r = requests.get(url=url, stream=True)
    with open(folder + '/' + tag + '.jpg', 'wb') as file:
        for chunk in r:
            file.write(chunk)


def google_search(search_term,
                  api_key='AIzaSyBS2lkbK22yeNN5dpnnUWRydtBsUient3s',
                  cse_id='017770384747286092867:4l4w65bmtkq', layer='items',
                  number=1, image=0, start_index=1, **kwargs):
    """
        Performs a google search based on the search term on either webpage(s) or image(s)
    :param search_term: Search of interest [String]
    :param api_key: Google Custom Search API key [String]
    Obtained from https://developers.google.com/custom-search/json-api/v1/overview
    :param cse_id: Your Google Custom Search's search engine ID [String]
    Obtained from cse.google.com > setup > Search engine ID
    :param layer: First layer to traverse for JSON result [String]
    View https://developers.google.com/custom-search/json-api/v1/reference/cse/list#parameters > Response
    :param image: search for image if true (Boolean)
    :param number: Number of search resulst to return [Integer]
    :param kwargs: Additional parameters such as type of search (webpage or image), image size, etc.
    More on https://developers.google.com/custom-search/json-api/v1/reference/cse/list#parameters > Parameters
    :return: List of dictionaries. Each search result resides in a dictionary
    """
    service = build('customsearch', 'v1', developerKey=api_key)

    if image:
        res = service.cse().list(q=search_term, cx=cse_id, num=number, searchType='image', start=start_index, **kwargs).execute()
        return res['items']
    else:
        res = service.cse().list(q=search_term, cx=cse_id, num=number, start=start_index, **kwargs).execute()
        return res[layer]


if __name__ == "__main__":

    # scrape_for_scraps(site='',
    #                   html_regex_filter=r"^article-crumbs clearfix group-letter letter.*", iterate_trimmed=1,
    #                   nested_string_tag='h2',
    #                   csv_header=['doggo_breed'], output_csv='breeds')

    with open('breeds.csv', 'r') as file:
        reader = csv.reader(file)
        breed_list = [breed[0] for breed in list(reader)]

    # for breed in breed_list:
    #     i = 1
    #     while i < 46:
    #         print('~~~ Doing Image Search for ' + breed + ' - Images ' + str(i) + ' to ' + str(i+8) + ' ~~~')
    #         image_search = google_search(breed, number=9, image=1, start_index=i)
    #         links = [result['link'] for result in image_search]
    #
    #         for num, link in enumerate(links):
    #             print('~~~ Start downloading for ' + breed + ' - Image #' + str(i+num) + ' ~~~')
    #             try:
    #                 photo_downloader(url=link, tag=breed + '_' + str(i+num-1), folder='images/')
    #             except Exception:
    #                 continue
    #             print('~~~ Download complete for ' + breed + ' - Image #' + str(i+num) + ' ~~~')
    #         i = i + 9