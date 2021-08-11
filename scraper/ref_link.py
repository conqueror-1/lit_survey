import pandas as pd
import json,requests

#The purpose of the script is to find links to all the papers 

DATA_PATH = "~/Projects/w2v/data/dataset.csv"
OUTPUT_FILE_PATH ="~/Projects/w2v/data/links.csv"

#RapidAPI google search API key
#https://rapidapi.com/apigeek/api/google-search3 (make an account here)
RAPID_API_KEY = "76a7736418msh1b468d99d3f2fe4p135c8fjsnca85576fc8ac"


data = pd.read_csv (DATA_PATH)
papers = data['PAPERS'].tolist()
df = pd.DataFrame (papers)
df.index += 1

######################
#  get references  #
######################

def return_clean_string(arg1):
    """TODO: Docstring for return_clean_string.

                    :arg1: pandas dataframe 
                    :returns: 

                    """
    pass
def get_title_from_ref(ref):
    """TODO: Docstring for get_title_from_ref.

                    :ref: paper reference string. 
                    :returns:clean paper reference string. 

                    """
    return(
        max(
            ref.split('.'), key=len)
            .strip().replace('"', '').replace("'", '')
    )

    
def create_request_url(title):
    """Replaces space characters with '+' to form a suitable query string for the API"""
    q_string = title.replace(' ', '+')
    return f"https://google-search3.p.rapidapi.com/api/v1/search/q={q_string}num=2"

def get_request_data(i, title):
    """Retrieves a link for a given title from the references.
            
            - if no link is found: return 'no link found'
            - if error on req:     return 'request failed'
    """
    
    headers = {
        'x-rapidapi-key': RAPID_API_KEY,
        'x-rapidapi-host': "google-search3.p.rapidapi.com"
    }
    
    query_s = create_request_url(title)
    
    link = ""

    # if you want a verbose output for each link, uncomment the print statements
    # print(f"Getting link for [{i}]: {title[:20]}...")
    
    try:
        r = requests.request("GET", query_s, headers=headers)
    except ConnectionError:
        pass
        
    if r.status_code == 200:
        j = json.loads(r.text)
        try:
            link += j['results'][0]['link']
        except:
            link += 'no link found'
    else:
        link += 'request failed'

    # print(f"Done: [{link}]")

    return link

df['title'] = df[0].apply (lambda x : get_title_from_ref(x))

links = []
for i, j in df.iterrows():
    link = get_request_data(i, j['title'])
    links.append(link)

df['link'] = links

# save the output file with the links
df.index += 1
df.to_csv(OUTPUT_FILE_PATH)
