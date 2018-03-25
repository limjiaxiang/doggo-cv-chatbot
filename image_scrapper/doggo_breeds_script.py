import requests
from bs4 import BeautifulSoup


def scrape_doggo_breeds(doggo_site):
    lst = []

    # use requests get function to obtain the article's webpage
    request = requests.get(doggo_site)

    # convert article webpage object to html
    data = request.text

    # convert the html to a BeautifulSoup object (nested data structure)
    soup = BeautifulSoup(data, 'html.parser')

    # View HTML code of article webpage in BeautifulSoup format
    print(soup.prettify())

if __name__ == "__main__":
    scrape_doggo_breeds('http://dogtime.com/dog-breeds/profiles')