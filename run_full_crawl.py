
import asyncio
import os
import json
import pandas as pd
from scraper import NESDCScraper, OUTPUT_CSV

STATUS_FILE = "data/metadata/collection_status.json"
TARGET_CSV = "data/metadata/polls_metadata_full.csv"

def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {"last_page": 1}

def save_status(page):
    with open(STATUS_FILE, "w") as f:
        json.dump({"last_page": page}, f)

def load_existing_ids():
    if os.path.exists(TARGET_CSV):
        try:
            df = pd.read_csv(TARGET_CSV)
            if "nttId" in df.columns:
                return set(df["nttId"].astype(str)) # compare as strings
        except Exception as e:
            print(f"Error loading existing metadata: {e}")
    return set()

async def main():
    print("Starting FULL CRAWL...")
    scraper = NESDCScraper(output_csv=TARGET_CSV)
    
    status = load_status()
    start_page = status.get("last_page", 1)
    
    existing_ids = load_existing_ids()
    print(f"Loaded {len(existing_ids)} existing records.")

    async with scraper.get_playwright_context() as page:
        scraper.page = page
        
        current_page = start_page
        consecutive_empty_pages = 0

        while True: # Loop until no posts found
            print(f"\n--- Processing Page {current_page} ---")
            
            # Get posts
            ids = await scraper.get_posts_on_page(current_page)
            
            if not ids:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 3:
                     print("No posts found for 3 consecutive pages. Stopping.")
                     break
                print("Empty page, checking next...")
                current_page += 1
                continue
            else:
                consecutive_empty_pages = 0

            # Process posts
            for ntt_id in ids:
                # Check duplication
                if str(ntt_id) in existing_ids:
                    print(f"Skipping {ntt_id} (Already exists).")
                    continue
                
                info = await scraper.process_post(ntt_id)
                if info:
                    scraper.append_data(info)
                    existing_ids.add(str(ntt_id))
            
            # Save status after successfull page
            save_status(current_page + 1)
            current_page += 1
            
    print("Full Crawl Completed.")

if __name__ == "__main__":
    asyncio.run(main())
