
import asyncio
import os
import pandas as pd
from scraper import NESDCScraper, OUTPUT_CSV

TARGET_CSV = "data/metadata/polls_metadata_daily_clean.csv"
MAX_PAGES_SAFETY = 10  # Hard limit to prevent infinite run if DB is empty

def load_existing_db():
    """Returns a set of nttIds from daily CSV."""
    db = set()
    if os.path.exists(TARGET_CSV):
        try:
            df = pd.read_csv(TARGET_CSV)
            if "nttId" in df.columns:
                db.update(df["nttId"].astype(str).tolist())
        except Exception as e:
            print(f"Error loading {TARGET_CSV}: {e}")
    return db

async def main():
    print("Starting DAILY UPDATE...")
    print("Scanning for new posts...")
    
    scraper = NESDCScraper(output_csv=TARGET_CSV)
    
    existing_ids = load_existing_db()
    print(f"Loaded {len(existing_ids)} existing records from Daily CSV.")

    async with scraper.get_playwright_context() as page:
        scraper.page = page
        
        page_num = 1
        MIN_PAGES = 5  # Check at least 50 items (5 pages)
        total_new_items = 0
        
        while True:
            # Safety break to prevent infinite loops if something goes wrong
            if page_num > 50: 
                print("Reached safety limit of 50 pages. Stopping.")
                break

            print(f"\n--- Scanning Page {page_num} ---")
            ids = await scraper.get_posts_on_page(page_num)
            
            if not ids:
                print("No more posts found.")
                break

            processed_on_page = 0
            
            for ntt_id in ids:
                str_id = str(ntt_id)
                
                # Check if it's already in our local CSV
                if str_id in existing_ids:
                    # It exists, but we might check if we have the file? 
                    # The user said "embargoes release sequentially? No, sequentially release NOT garuanteed."
                    # Users concern: "Even if 5 consecutive are collected, a later one might be uncollected."
                    # But if `existing_ids` contains it, we already processed it successfully (including file download attempt).
                    # If we failed to download the file previously (embargoed), we might not have added it to `existing_ids`?
                    # Let's check `process_post` return value and `existing_ids` logic.
                    # Currently: `if info: scraper.append_data(info); existing_ids.add(str_id)`
                    # So `existing_ids` ONLY contains completed posts.
                    # Embargoed posts are NOT in `existing_ids`.
                    # So we WILL try to process them again if we encounter them. Correct.
                    print(f"Skipping {ntt_id} (Already collected)")
                    continue
                
                # Not in DB -> Try to process (New or previously Embargoed)
                print(f"Checking {ntt_id}...")
                info = await scraper.process_post(ntt_id)
                
                if info:
                    # Success! We found 'collectable data'.
                    scraper.append_data(info)
                    existing_ids.add(str_id)
                    processed_on_page += 1
                else:
                    # Embargoed or error. 
                    # Does this count as "collected data" for extension purposes?
                    # User said: "If data to collect appears" -> Implies successful collection.
                    # If it remains embargoed, we didn't "collect" it.
                    pass

            print(f"Page {page_num} result: {processed_on_page} new items collected.")
            total_new_items += processed_on_page

            # Termination Logic
            # 1. Must scan at least MIN_PAGES (5).
            # 2. If we are past MIN_PAGES, we only continue if the LAST page had collected data.
            #    (User: "If data comes out in the last 10, extend by 10")
            
            if page_num < MIN_PAGES:
                # Keep going until we hit minimum
                page_num += 1
                continue
            else:
                # We are at or past the minimum.
                if processed_on_page > 0:
                    print(f"Found {processed_on_page} items on page {page_num}. Extending scan range...")
                    page_num += 1
                    continue
                else:
                    print(f"No new items on page {page_num} (and satisfied min {MIN_PAGES} pages). Stopping.")
                    break
            
        print("Daily Update Cycle Completed.")
        
        # Windows Notification
        try:
            from plyer import notification
            notification_title = "여론조사 수집기 (NESDC Scraper)"
            notification_message = f"신규 {total_new_items}건 수집 완료."
            
            notification.notify(
                title=notification_title,
                message=notification_message,
                app_name='NESDC Scraper',
                timeout=10
            )
            print("Windows notification sent.")
        except Exception as e:
            print(f"Failed to send notification: {e}")

if __name__ == "__main__":
    asyncio.run(main())
