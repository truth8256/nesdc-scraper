
import os
import re
import asyncio
import pandas as pd
from contextlib import asynccontextmanager
from playwright.async_api import async_playwright

# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
BASE_URL = "https://www.nesdc.go.kr/portal/bbs/B0000005/list.do?menuNo=200467"
DATA_DIR = "./data/raw"
OUTPUT_CSV = "./data/metadata/polls_metadata.csv"

# ─────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────
def sanitize_filename(s: str) -> str:
    """Sanitize filename to be safe for Windows file system."""
    return re.sub(r'[\\/*?:"<>|]', "_", s or "").strip()[:180]

async def safe_text(element, selector):
    """Safe text extraction from a locator."""
    try:
        if await element.locator(selector).count() > 0:
            text = await element.locator(selector).first.inner_text()
            return re.sub(r'\s+', ' ', text).strip()
        return ""
    except:
        return ""

class NESDCScraper:
    def __init__(self, output_csv=OUTPUT_CSV):
        self.collected_data = []
        self.page = None
        self.output_csv = output_csv
        # Ensure base directories exist
        os.makedirs(DATA_DIR, exist_ok=True)

    @asynccontextmanager
    async def get_playwright_context(self):
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(accept_downloads=True)
            page = await context.new_page()
            # Handle unexpected dialogs
            page.on("dialog", lambda dialog: asyncio.create_task(dialog.dismiss()))
            yield page
            await browser.close()

    async def run(self):
        async with self.get_playwright_context() as page:
            self.page = page

            print(f"Starting NESDC Scraper (Output: {self.output_csv})...")
            
            # 1. Get List of Posts to Scrape
            # For now, let's start with page 1 to get the latest posts.
            # In a full run, we might want to iterate pages or check against existing DB.
            target_ids = await self.get_latest_posts_ids(pages_to_scrape=3) # Scrape first 3 pages for demo
            
            print(f"Found {len(target_ids)} posts to process.")

            # 2. Process each post
            for nttid in target_ids:
                await self.process_post(nttid)

            # 3. Save Results
            self.save_to_csv()

    async def get_posts_on_page(self, page_num):
        """Scrapes a specific list page to get post IDs."""
        ids = []
        url = f"{BASE_URL}&pageIndex={page_num}"
        print(f"Scanning list page {page_num}...")
        try:
            await self.page.goto(url, wait_until="networkidle", timeout=20000)
            
            # Selector verified: div.board div.grid a.row.tr
            links = self.page.locator("a.row.tr")
            count = await links.count()
            
            if count == 0:
                print(f"No posts found on page {page_num}.")
                return []
            
            for i in range(count):
                href = await links.nth(i).get_attribute("href")
                if href and "nttId=" in href:
                    match = re.search(r"nttId=(\d+)", href)
                    if match:
                        ids.append(int(match.group(1)))
            
            print(f"   -> Found {len(ids)} posts on page {page_num}.")
            return ids
        except Exception as e:
            print(f"Error scanning page {page_num}: {e}")
            return []

    async def process_post(self, nttid):
        """Processes a single post: extracts metadata and downloads files."""
        url = f"https://www.nesdc.go.kr/portal/bbs/B0000005/view.do?nttId={nttid}"
        print(f"Processing nttId={nttid}...")

        try:
            await self.page.goto(url, wait_until="networkidle", timeout=30000)

            # Check for "Analysis Result" file availability
            analysis_row = self.page.locator("tr", has=self.page.locator("th", has_text="결과분석 자료"))
            analysis_file_link = analysis_row.locator("a.ico_pdf").first
            
            has_analysis = await analysis_file_link.count() > 0
            if not has_analysis:
                print(f"Skipping {nttid}: No 'Analysis Result' file found (likely embargoed).")
                return None

            # Basic Metadata
            idx = await safe_text(self.page, 'xpath=//th[contains(text(), "등록 글번호")]/following-sibling::td')
            if not idx:
                print(f"Skipping {nttid}: Could not find Registration Number (idx).")
                return None

            info = {
                "nttId": nttid,
                "idx": idx,
                "req": "", "org": "", "org_joint": "", "elect": "", "area": "", "contest": "",
                "date": "", "n": "", "cont_rate": "", "resp_rate": ""
            }

            async def get_val(header_text):
                # Use strict equality for header text to avoid partial matches (e.g. "조사의뢰자" matching "조사의뢰자 URL")
                return await safe_text(self.page, f'xpath=//th[normalize-space(text())="{header_text}"]/following-sibling::td')

            # Extract fields
            raw_req = await get_val("조사의뢰자")
            if ":" in raw_req:
                info["req"] = raw_req.split(":", 1)[-1].strip()
            else:
                info["req"] = raw_req.strip()

            info["org"] = await get_val("조사기관명")
            info["org_joint"] = await get_val("공동조사기관명")
            info["elect"] = await get_val("선거구분")
            info["area"] = await get_val("지역")
            info["contest"] = await get_val("선거명")
            info["date"] = await get_val("조사일시")
            info["n"] = await safe_text(self.page, 'xpath=//th[contains(., "조사완료 사례수(명)")]/../following-sibling::tr[contains(@class,"th")]/td[1]')
            if not info["n"]:
                 info["n"] = await safe_text(self.page, 'xpath=//tr[th[contains(normalize-space(.), "응답완료 사례수")]]/td[1]')

            info["cont_rate"] = await safe_text(self.page, 'xpath=//th[normalize-space(text())="전체 접촉률"]/following-sibling::td')
            info["resp_rate"] = await safe_text(self.page, 'xpath=//th[normalize-space(text())="전체 응답률"]/following-sibling::td')

            # Survey Methods (Set 1-5)
            for i in range(1, 6):
                blk = self.page.locator(f'div.set{i}')
                if await blk.count() == 0: continue
                if not await blk.is_visible(): continue

                t1 = blk.locator("table.view.ex").nth(0)
                info[f"method{i}"] = await safe_text(t1, "tr:first-child td")
                info[f"method_rate{i}"] = await safe_text(t1, "tr:nth-child(2) td")
                
                t2 = blk.locator("table.view.ex").nth(1)
                info[f"frame{i}"] = await safe_text(t2, "xpath=.//th[contains(., '추출틀')]/following-sibling::td")

            # File Downloads
            save_dir = os.path.join(DATA_DIR, sanitize_filename(idx))
            os.makedirs(save_dir, exist_ok=True)

            q_row = self.page.locator("tr", has=self.page.locator("th", has_text="전체질문지 자료"))
            q_link = q_row.locator("a.ico_pdf").first
            if await q_link.count() > 0:
                info["qnaire_file"] = await self.download_file(q_link, save_dir, idx, "전체질문지")
            else:
                info["qnaire_file"] = ""

            info["analysis_file"] = await self.download_file(analysis_file_link, save_dir, idx, "결과분석")

            return info

        except Exception as e:
            print(f"❌ Error processing {nttid}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def append_data(self, data):
        """Appends a single data record to the CSV file (Thread-safe logic required if parallel, but here sequential)."""
        # Define strict column order
        columns = [
            "nttId", "idx", "req", "org", "org_joint", "elect", "area", "contest", "date", "n",
            "cont_rate", "resp_rate",
            "method1", "method_rate1", "frame1",
            "method2", "method_rate2", "frame2",
            "method3", "method_rate3", "frame3",
            "method4", "method_rate4", "frame4",
            "method5", "method_rate5", "frame5",
            "qnaire_file", "analysis_file"
        ]
        
        df = pd.DataFrame([data])
        
        # Ensure all columns exist with empty string default
        for col in columns:
            if col not in df.columns:
                df[col] = ""
                
        # Enforce order
        df = df[columns]

        if os.path.exists(self.output_csv):
            df.to_csv(self.output_csv, mode='a', header=False, index=False, encoding='utf-8-sig')
        else:
            df.to_csv(self.output_csv, mode='w', header=True, index=False, encoding='utf-8-sig')
            
        print(f"Saved record {data.get('idx', 'unknown')} to {self.output_csv}.")

    async def get_latest_posts_ids(self, pages_to_scrape=1):
        """(Deprecated) wrapper for backward compatibility."""
        all_ids = []
        for p in range(1, pages_to_scrape + 1):
            ids = await self.get_posts_on_page(p)
            all_ids.extend(ids)
        return list(dict.fromkeys(all_ids))


    async def download_file(self, link_locator, save_dir, idx, prefix):
        try:
            async with self.page.expect_download() as dl_info:
                await link_locator.click()
            dl = await dl_info.value
            
            # Use original filename
            original_name = dl.suggested_filename
            # Sanitize
            safe_name = sanitize_filename(original_name)
            save_path = os.path.join(save_dir, safe_name)
            
            # Avoid overwrite collision
            if os.path.exists(save_path):
                base, ext = os.path.splitext(save_path)
                counter = 2
                while True:
                    new_path = f"{base}_{counter}{ext}"
                    if not os.path.exists(new_path):
                        save_path = new_path
                        safe_name = os.path.basename(save_path)
                        break
                    counter += 1
            
            await dl.save_as(save_path)
            print(f"   -> Downloaded: {safe_name}")
            return safe_name
        except Exception as e:
            print(f"   Warning: Download failed: {e}")
            return ""

    def save_to_csv(self):
        if not self.collected_data:
            print("⚠️ No data collected.")
            return

        df = pd.DataFrame(self.collected_data)
        
        # Ensure column order matches requirements
        columns = [
            "nttId", "idx", "req", "org", "org_joint", "elect", "area", "contest", "date", "n",
            "cont_rate", "resp_rate",
            "method1", "method_rate1", "frame1",
            "method2", "method_rate2", "frame2",
            "method3", "method_rate3", "frame3",
            "method4", "method_rate4", "frame4",
            "method5", "method_rate5", "frame5",
            "qnaire_file", "analysis_file"
        ]
        
        # Add missing columns with empty string
        for col in columns:
            if col not in df.columns:
                df[col] = ""
        
        # Reorder and save
        df = df[columns]
        
        # Append if exists? No, clarify requirement. Usually "save to" implies over-write or append. 
        # For a scraper, appending is safer if run multiple times, but let's just write fresh for now 
        # as per user request to "save results".
        output_path = OUTPUT_CSV # Save to root as requested or data dir? User said "polls_metadata.csv"
        # Let's save to current dir as per user request implicit path
        
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"💾 Saved {len(df)} records to {output_path}")

if __name__ == "__main__":
    scraper = NESDCScraper()
    asyncio.run(scraper.run())
