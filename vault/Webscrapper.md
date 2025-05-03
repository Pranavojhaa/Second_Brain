**Current Progress 25th April 2025**
	Tech Stack
		1. Python 
		2. Open AI 
		3. Chroma DB  
		4. LangChain
		5. StreamLit 
	Libraries
		6. Requests 
		7. BeautifulSoup 
		8. Selenium 
		9. Time 
		10. Pandas 
		11. LangChain 
	Screens
		Main.py
			logic - This Python script is designed as a tool for web scraping, utilizing Selenium to handle HTML parsing on potentially complex or dynamic websites. The script begins by prompting the user to input a URL, ensuring it's in an executable format (adding "https://" if necessary). It then employs `get_soup` to parse the webpage into a BeautifulSoup object, validating the structure with `validate_soup`. If validation fails, it prompts the user to retry with Selenium, likely handling JavaScript and frames. Once validated successfully ("OK"), the script proceeds to fetch more detailed element data using `fetch_page`, which is displayed for inspection. Finally, it allows interactive selection of elements via `inspect_element` before exiting gracefully if any issues arise during the process. The tool combines basic scraping capabilities with an interactive inspection feature, making it suitable for users seeking a straightforward approach to web scraping and element analysis.
			Libraries - We are calling everything from already definded functions from above pages 
		Inspector.py
			logic - Inspector.py focuses on fetching the page, then extracting the soup for that page (HTML Code) then using the the find_all function to find different html attributes in the soup such as: 
				</table> = Table 
				</p> = Paragraphs 
				</img> = Images 
				</h> = Headings 
				</a> = Links (Href)
			Find_all return everything as a list so then using the python in built funcntion len() - Find how many of each element is there and out to the user, this page then takes user input on which element the user wants to chose, depending on the choice of the user the "inspect_element" calls all the functions from our scraper.py to display elements, Then it goes to processing which again takes input from the user, what to do with the scraped element: 
				1. Inspect Another element 
				2. Export to the choose file format (Under Work)
				3. AI Analyis (Under Work)
				4. Exit out of the scrapper 
			Libraries - Requests, BeautifulSoup, All functions of Scraper.py
		Scraper.py 
			Logic - ```
```
def fetch_page(url):
	response = requests.get(url)
	soup = BeautifulSoup(response, 'html.parser')
	table = len(soup.find_all('table'))
	paragraphs = len(soup.find_all('p'))
	headings = len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']))
	images = len(soup.find_all('img'))
	links = len(soup.find_all('a')) 
	return {

	'table': table,
	'paragraphs': paragraphs,
	'headings': headings,
	'images': images,
	'links': links,
	'quotes': quotes
	}
```


**Future Plans**
	Ideation: 
	




```
```