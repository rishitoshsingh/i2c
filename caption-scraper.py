from bs4 import BeautifulSoup
import csv

# Open the HTML file
with open('flickr30k.html', 'r') as file:
    html_code = file.read()

soup = BeautifulSoup(html_code, 'html.parser')

# Find all table rows
rows = soup.find_all('tr')

# Create a CSV file to store the scraped data
csv_file = open('flickr30k-captions.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['image_id', 'caption'])

# Scrape image names and captions
failed = 0
running_image_id = None

for row in rows:
    # try:
    is_image_id_row = row.find('a')
    if is_image_id_row == None:
        captions = [caption.text.strip() for caption in row.find_all("li")]
        for caption in captions:
            csv_writer.writerow([running_image_id, caption])
    else:
        running_image_id = is_image_id_row["href"].replace(".jpg","")
csv_file.close()

import pandas as pd
data = pd.read_csv("flickr30k-captions.csv")
data = data.sample(frac=1)
data.to_csv("flickr30k-captions.csv", index=False)