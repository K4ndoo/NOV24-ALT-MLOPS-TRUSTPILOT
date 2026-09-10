import pandas as pd
import json
import time
import random
import os
import argparse
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

# Topics to scrape
TOPICS = {
    'customer_service': 'service client',
    'delivery_service': 'service livraison',
    'product': 'qualité produit'
}

COMPANIES = [
    {'domain': 'www.showroomprive.com', 'name': 'ShowRoom'},
    {'domain': 'www.veepee.fr', 'name': 'VeePee'}
]

DATASET_PATH = 'data/labelled_dataset/dataset_avis.csv'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_pages', type=int, default=5, help='Maximum number of pages to scrape per category')
    args = parser.parse_args()
    
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        df['comment_id'] = df['comment_id'].astype(str)
        existing_ids = set(df['comment_id'].unique())
        previously_existing_ids = set(df['comment_id'].unique())
        initial_count = len(existing_ids)
    else:
        columns = [
            'comment_id', 'Commentaire', 'star', 'date', 'client', 'reponse', 
            'source', 'company', 'ville', 'maj', 'date_commande', 'ecart', 
            'clean_comment', 'qualité produit', 'service livraison', 'service client'
        ]
        df = pd.DataFrame(columns=columns)
        existing_ids = set()
        previously_existing_ids = set()
        initial_count = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for company_info in COMPANIES:
            domain = company_info['domain']
            company_name = company_info['name']
            print(f"Scraping company: {company_name}")
            
            for topic_url, topic_label in TOPICS.items():
                print(f"  Scraping topic: {topic_label}")
                for page_num in range(1, args.max_pages + 1):

                    # anti bot (on délai un peu)
                    sleep_time = random.uniform(14.0, 32.0)
                    print(f"  Sleeping for {sleep_time:.2f} seconds...")
                    time.sleep(sleep_time)

                    url = f"https://fr.trustpilot.com/review/{domain}?page={page_num}&topics={topic_url}"
                    print(f"  Fetching {url}...")
                    
                    try:
                        # wait_until="networkidle" bypasses the WAF JS challenge by waiting for it to finish
                        response = page.goto(url, wait_until='networkidle', timeout=45000)
                        
                        if response and response.status == 404:
                            print("  Reached 404, stopping pagination for this topic.")
                            break
                            
                        html_content = page.content()
                    except Exception as e:
                        print(f"  Error fetching {url}: {e}")
                        break
                        
                    soup = BeautifulSoup(html_content, 'html.parser')
                    script_tag = soup.find('script', id='__NEXT_DATA__', type='application/json')
                    
                    if not script_tag:
                        print("  Could not find __NEXT_DATA__ script tag (anti-bot might have blocked).")
                        break
                        
                    try:
                        json_data = json.loads(script_tag.string)
                        reviews = json_data.get('props', {}).get('pageProps', {}).get('reviews', [])
                    except Exception as e:
                        print(f"  Error parsing JSON: {e}")
                        break
                        
                    if not reviews:
                        print("  No reviews found on this page.")
                        break
                        
                    all_already_in_csv = True
                    new_rows = []
                    
                    for r in reviews:
                        c_id = str(r.get('id', ''))
                        text = r.get('text', '')
                        
                        # On ignore les avis sans texte (inutiles pour notre modèle NLP)
                        if not c_id or not text or str(text).strip() == "":
                            continue
                            
                        # Si l'avis n'était pas dans le CSV AVANT le lancement de ce script, 
                        # on ne doit pas arrêter la pagination.
                        if c_id not in previously_existing_ids:
                            all_already_in_csv = False
                            
                        if c_id in existing_ids:
                            idx = df.index[df['comment_id'] == c_id]
                            if not idx.empty:
                                df.loc[idx, topic_label] = 1
                        else:
                            text = r.get('text', '')
                            rating = r.get('rating', '')
                            dates = r.get('dates', {})
                            published_date = dates.get('publishedDate', '')
                            consumer = r.get('consumer', {})
                            client = consumer.get('displayName', '')
                            
                            new_row = {
                                'comment_id': c_id,
                                'Commentaire': text,
                                'star': rating,
                                'date': published_date,
                                'client': client,
                                'reponse': '',
                                'source': 'TrustPilot_Scrap',
                                'company': company_name,
                                'ville': '',
                                'maj': '',
                                'date_commande': '',
                                'ecart': '',
                                'clean_comment': '',
                                'qualité produit': 0,
                                'service livraison': 0,
                                'service client': 0
                            }
                            new_row[topic_label] = 1
                            new_rows.append(new_row)
                            existing_ids.add(c_id)
                    
                    if new_rows:
                        new_df = pd.DataFrame(new_rows)
                        df = pd.concat([df, new_df], ignore_index=True)
                        
                        df['qualité produit'] = pd.to_numeric(df['qualité produit'], errors='coerce').fillna(0).astype(int)
                        df['service livraison'] = pd.to_numeric(df['service livraison'], errors='coerce').fillna(0).astype(int)
                        df['service client'] = pd.to_numeric(df['service client'], errors='coerce').fillna(0).astype(int)
                        
                        df.to_csv(DATASET_PATH, index=False)
                        
                    if all_already_in_csv:
                        print("  All reviews on this page were already in the original dataset. Stopping pagination.")
                        break

        browser.close()

    df['qualité produit'] = pd.to_numeric(df['qualité produit'], errors='coerce').fillna(0).astype(int)
    df['service livraison'] = pd.to_numeric(df['service livraison'], errors='coerce').fillna(0).astype(int)
    df['service client'] = pd.to_numeric(df['service client'], errors='coerce').fillna(0).astype(int)
    
    agg_funcs = {col: 'first' for col in df.columns}
    agg_funcs['qualité produit'] = 'max'
    agg_funcs['service livraison'] = 'max'
    agg_funcs['service client'] = 'max'
    del agg_funcs['comment_id']
    
    df = df.groupby('comment_id', as_index=False).agg(agg_funcs)
    df.to_csv(DATASET_PATH, index=False)
    final_count = len(df)
    print(f"Scraping finished. {final_count - initial_count} new reviews added. Total reviews: {final_count}. Data saved.")

if __name__ == "__main__":
    main()
