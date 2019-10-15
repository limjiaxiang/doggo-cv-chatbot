import re
import csv
import os
import time

import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
import cv2

from image_preprocess import resize_image


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


def photo_downloader(url, tag, folder, resize=None):
    r = requests.get(url=url, stream=True)
    image_filepath = os.path.join(folder, '.'.join([tag, 'jpg']))
    with open(image_filepath, 'wb') as image:
        for chunk in r:
            image.write(chunk)
    if resize:
        img = cv2.imread(image_filepath)
        resized_img = resize_image(img, pad=False, stretch=True, resized_width=resize[0], resized_height=resize[1])
        cv2.imwrite(image_filepath, resized_img)


def google_search(search_term, api_key, cse_id,
                  layer='items', number=1, image=0, start_index=1, **kwargs):
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
    :param number: Number of search results to return [Integer]
    :param kwargs: Additional parameters such as type of search (webpage or image), image size, etc.
    More on https://developers.google.com/custom-search/json-api/v1/reference/cse/list#parameters > Parameters
    :return: List of dictionaries. Each search result resides in a dictionary
    """
    service = build('customsearch', 'v1', developerKey=api_key)

    if image:
        res = service.cse().list(q=search_term, cx=cse_id, num=number, searchType='image',
                                 start=start_index, **kwargs).execute()
        return res['items']
    else:
        res = service.cse().list(q=search_term, cx=cse_id, num=number, start=start_index, **kwargs).execute()
        return res[layer]


if __name__ == "__main__":

    # scrape_for_scraps(site='',
    #                   html_regex_filter=r"^article-crumbs clearfix group-letter letter.*", iterate_trimmed=1,
    #                   nested_string_tag='h2',
    #                   csv_header=['doggo_breed'], output_csv='breeds')

    with open('search_engine_creds.json') as creds_json:
        cred_dict = eval(creds_json.read())
        api_key = cred_dict['api_key']
        cse_id = cred_dict['cse_id']

    with open('breeds temp.csv', 'r') as file:
        reader = csv.reader(file)
        breed_list = [breed[0] for breed in list(reader)][1:]

    for breed in breed_list:
        i = 1

        breed_dir = os.path.join('../images/' + breed)

        if not os.path.isdir(breed_dir):
            os.makedirs(breed_dir)

        search_breed = breed
        if 'dog' not in breed:
            search_breed = ' '.join([breed, 'dog'])

        while i < 100:
            print('~~~ Doing Image Search for ' + breed + ' - Images ' + str(i) + ' to ' + str(i+9) + ' ~~~')
            image_search = google_search(search_breed, api_key=api_key, cse_id=cse_id, number=10, image=1, start_index=i)
            links = [result['link'] for result in image_search]

            for num, link in enumerate(links):
                print('~~~ Start downloading for ' + breed + ' - Image #' + str(i+num) + ' ~~~')
                try:
                    photo_downloader(url=link, tag=search_breed + '_' + str(i+num), folder=breed_dir,
                                     resize=(250, 250))
                except Exception as e:
                    print(e)
                    print('Did not download for', breed, 'photo number', i+num)
                    continue
                print('~~~ Download complete for    ' + breed + ' - Image #' + str(i+num) + ' ~~~')
            i = i + 10
            time.sleep(1)
