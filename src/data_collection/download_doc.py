"""
Download images from TDTU documents and convert to PDF
"""
import os
import json
import time
import re
from typing import List, Optional
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from tdtu_client import TDTUClient
import base64
import hashlib

load_dotenv()

class ImageToPDFConverter:
    """Download images from document pages and convert to PDF"""
    def __init__(self, output_dir: str = "downloads_pdf"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def sanitize_filename(self, filename: str) -> str:
        """Remove invalid characters from filename"""
        # Remove invalid characters
        filename = re.sub(r'[<>:"\\|?*]', '', filename)
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        return filename.strip()
    
    def download_images_from_page(self, client: TDTUClient, url: str, title: str) -> List[str]:
        logger.info(f" Opening: {title[:80]}...")
        
        # CRITICAL: Clear selenium-wire request cache BEFORE loading new document
        logger.info("🗑️  Clearing request cache...")
        try:
            del client.driver.requests
        except:
            pass
        
        # Navigate to document page with verification
        logger.info(f"Navigating to: {url}")
        client.driver.get(url)
        
        # Wait and verify navigation succeeded
        max_retries = 2
        for attempt in range(max_retries):
            time.sleep(3)
            current_url = client.driver.current_url
            
            if 'Detail' in current_url and current_url != 'https://quychehocvu.tdtu.edu.vn':
                logger.info(f"Navigation successful: {current_url}")
                break
            else:
                if attempt < max_retries - 1:
                    logger.info(f"🔄 Retrying navigation...")
                    client.driver.get(url)
                else:
                    logger.error(f"Failed to navigate to document after {max_retries} attempts")
                    return []
        
        time.sleep(5) 
        
        # Wait for PDF.js to load
        logger.info(" Waiting for PDF.js to load...")
        time.sleep(5)
        
        # CRITICAL: Reset scroll position to TOP before extracting
        logger.info("Resetting scroll position to top...")
        client.driver.execute_script("""
            let container = document.querySelector('#viewerContainer, .viewerContainer, [class*="viewer-container"], .pdfViewer');
            if (container) {
                container.scrollTop = 0;
                console.log('Scroll reset to 0');
            }
        """)
        time.sleep(2)  # Wait for scroll to settle
        
        # Strategy: Scroll and extract incrementally
        logger.info("Incrementally scrolling and extracting pages...")
        
        # Get container info
        container_info = client.driver.execute_script("""
            let container = document.querySelector('#viewerContainer, .viewerContainer, [class*="viewer-container"], .pdfViewer');
            if (!container) return null;
            
            return {
                scrollHeight: container.scrollHeight,
                clientHeight: container.clientHeight
            };
        """)
        
        all_images_data = {}  # Use dict to avoid duplicates
        
        if container_info:
            logger.info(f"  Container height: {container_info['scrollHeight']}px")
            scroll_height = container_info['scrollHeight']
            client_height = container_info['clientHeight']
            step_size = client_height  # One viewport height per step
            
            position = 0
            step_num = 0
            
            # Start from TOP (position 0)
            logger.info(f"  Starting scroll extraction from position 0...")
            
            while position <= scroll_height:
                # Scroll to position
                client.driver.execute_script(f"""
                    let container = document.querySelector('#viewerContainer, .viewerContainer, [class*="viewer-container"], .pdfViewer');
                    if (container) container.scrollTop = {position};
                """)
                
                time.sleep(2)  # Wait for canvases to render
                
                # Extract visible canvases NOW (before they get destroyed)
                images_at_position = client.driver.execute_script("""
                    let images = [];
                    let canvases = document.querySelectorAll('canvas');
                    
                    for (let canvas of canvases) {
                        if (canvas.width > 100 && canvas.height > 100) {
                            try {
                                let dataUrl = canvas.toDataURL('image/png');
                                images.push(dataUrl);
                            } catch(e) {}
                        }
                    }
                    
                    return images;
                """)
                
                # Add to collection (using hash to avoid duplicates)
                for img_data in images_at_position:
                    
                    img_hash = hashlib.md5(img_data.encode()).hexdigest()
                    if img_hash not in all_images_data:
                        all_images_data[img_hash] = img_data
                
                step_num += 1
                logger.debug(f"    Step {step_num}: pos={position}, found {len(images_at_position)} canvases, total unique: {len(all_images_data)}")
                
                position += step_size
            
            logger.info(f"  Collected {len(all_images_data)} unique canvas images")
            png_data_urls = list(all_images_data.values())
        else:
            logger.warning("Could not find PDF viewer container")
            png_data_urls = []
        
        # Scroll back to top for next document
        logger.info("Resetting to top for next document...")
        client.driver.execute_script("""
            let container = document.querySelector('#viewerContainer, .viewerContainer, [class*="viewer-container"], .pdfViewer');
            if (container) {
                container.scrollTop = 0;
            }
        """)
        time.sleep(1)
        
        logger.info(f"Total collected: {len(png_data_urls)} canvas images")       
        
        # Create folder for this document
        safe_title = self.sanitize_filename(title)
        doc_folder = os.path.join(self.output_dir, safe_title)
        os.makedirs(doc_folder, exist_ok=True)
        
        downloaded_images = []
        
        # Save each canvas as PNG (all are data:image/png URLs)
        for i, data_url in enumerate(png_data_urls, 1):
            try:
                if not data_url.startswith('data:image/png;base64,'):
                    logger.warning(f"    ✗ Invalid data URL format")
                    continue
                
                # Extract base64 data
                base64_data = data_url.split('data:image/png;base64,')[1]
                
                # Decode base64 to binary
                
                image_data = base64.b64decode(base64_data)
                
                file_size = len(image_data) / 1024  # KB
                
                # Skip small images (likely empty/black canvases)
                if file_size < 50:  
                    logger.debug(f"  [{i}/{len(png_data_urls)}] Skipping small canvas ({file_size:.1f} KB)")
                    continue
                
                logger.info(f"  [{i}/{len(png_data_urls)}] Processing canvas PNG ({file_size:.1f} KB)...")
                
                # Save as PNG file 
                page_num = len(downloaded_images) + 1
                filename = f"page_{page_num:03d}.png"
                filepath = os.path.join(doc_folder, filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_data)
                
                logger.info(f"    ✓ Saved: {filename} ({file_size:.1f} KB)")
                
                downloaded_images.append(filepath)
                
            except Exception as e:
                logger.warning(f"    ✗ Failed to process canvas {i}: {e}")
                continue
        
        logger.info(f"Downloaded {len(downloaded_images)} images to: {doc_folder}")
        
        # Extract text content from network requests (RenderPdfPages API)
        logger.info("Extracting text content from RenderPdfPages API...")
        
        # Create text_data subfolder
        text_data_folder = os.path.join(doc_folder, "text_data")
        os.makedirs(text_data_folder, exist_ok=True)
        
        text_data = self.extract_text_from_network(client, text_data_folder)
        
        return downloaded_images
    
    def extract_text_from_network(self, client: TDTUClient, text_data_folder: str) -> list:

        try:
            # Wait a bit for all network requests to complete
            time.sleep(2)
            
            # Get all captured requests from selenium-wire
            all_pages_data = []
            
            for request in client.driver.requests:
                try:
                    # Get response body
                    if request.response:
                        response_body = request.response.body
                        if response_body:
                            # Decode and parse JSON
                            data = json.loads(response_body.decode('utf-8'))
                            
                            # Extract ONLY textContent and textBounds to reduce size
                            if 'textContent' in data or 'textBounds' in data:
                                page_data = {
                                    'textContent': data.get('textContent', []),
                                    'textBounds': data.get('textBounds', [])
                                }
                                all_pages_data.append(page_data)
                                logger.info(f"Captured RenderPdfPages data ({len(page_data['textContent'])} text items)")
                except Exception as e:
                    logger.debug(f" Failed to parse response: {e}")
                    continue
            
            if not all_pages_data:
                logger.warning(" No RenderPdfPages data captured from network")
                return []
            
            logger.info(f" Found {len(all_pages_data)} RenderPdfPages responses")
            
            # Save combined data
            combined_file = os.path.join(text_data_folder, "data.json")
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(all_pages_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"  Saved {len(all_pages_data)} pages to text_data/")
            return all_pages_data
            
        except Exception as e:
            logger.error(f"  Error extracting text data: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    
    def images_to_pdf(self, image_paths: List[str], output_pdf: str) -> bool:

        if not image_paths:
            logger.warning("No images to convert")
            return False
        
        try:
            logger.info(f"📄 Converting {len(image_paths)} images to PDF...")
            
            # Open all images and convert to RGB
            images = []
            for img_path in sorted(image_paths):  # Sort to maintain order
                try:
                    img = Image.open(img_path)
                    # Convert to RGB (PDF requires RGB mode)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to open image {img_path}: {e}")
                    continue
            
            if not images:
                logger.error("No valid images to convert")
                return False
            
            # Save as PDF
            first_image = images[0]
            other_images = images[1:] if len(images) > 1 else []
            
            first_image.save(
                output_pdf,
                save_all=True,
                append_images=other_images,
                resolution=100.0,
                quality=95,
                optimize=False
            )
            
            # Get file size
            pdf_size = os.path.getsize(output_pdf) / (1024 * 1024)  # MB
            logger.info(f" PDF created: {output_pdf} ({pdf_size:.2f} MB, {len(images)} pages)")
            
            # Close images
            for img in images:
                img.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create PDF: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def process_document(self, client: TDTUClient, doc_info: dict) -> Optional[str]:

        title = doc_info.get('title', 'Unknown')
        url = doc_info.get('url')
        
        if not url:
            logger.warning(f"No URL for document: {title}")
            return None
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing: {title}")
        logger.info(f"{'='*70}")
        
        # Download images
        image_paths = self.download_images_from_page(client, url, title)
        
        if not image_paths:
            logger.warning(f"No images downloaded for: {title}")
            return None
        
        # Create PDF
        safe_title = self.sanitize_filename(title)
        pdf_filename = f"{safe_title}.pdf"
        pdf_path = os.path.join(self.output_dir, pdf_filename)
        
        if self.images_to_pdf(image_paths, pdf_path):
            return pdf_path
        else:
            return None


def get_id_from_url(url: str) -> str:
    """Lấy ID từ cuối đường dẫn. Ví dụ: .../Detail/145 -> 145"""
    try:
        # Loại bỏ khoảng trắng và lấy phần tử cuối cùng sau dấu gạch chéo
        return url.strip().rstrip('/').split('/')[-1]
    except:
        return ""

def filter_documents(documents: List[dict]) -> List[dict]:
    
    # Hiển thị danh sách kèm ID để dễ nhìn
    print(f"\n📋 Tìm thấy {len(documents)} tài liệu.")
    print(f"{'STT':<5} | {'ID':<6} | {'Tên tài liệu'}")
    print("-" * 80)
    for i, doc in enumerate(documents[:10], 1): # Chỉ in 10 bài đầu làm mẫu
        doc_id = get_id_from_url(doc['url'])
        print(f"{i:<5} | {doc_id:<6} | {doc['title'][:60]}...")
    if len(documents) > 10:
        print(f"... và {len(documents)-10} tài liệu khác.")

    print("\nTÙY CHỌN TẢI:")
    print(" - Nhập 'all': Tải tất cả.")
    print(" - Nhập ID (vd: 145): Tải bài có ID 145.")
    print(" - Nhập danh sách ID (vd: 145, 143): Tải các bài có ID này.")
    print(" - Nhập từ khóa: Tìm theo tên.")
    
    choice = input("\n Nhập lựa chọn: ").strip()
    
    # 1. Tải tất cả
    if choice.lower() in ['all', '']:
        return documents

    selected_docs = []


    # 2. Xử lý danh sách ID rời rạc (ví dụ: 145, 143) hoặc 1 ID đơn lẻ
    # Tách dấu phẩy hoặc khoảng trắng
    input_ids = [x.strip() for x in choice.replace(',', ' ').split() if x.strip().isdigit()]
    
    if input_ids:
        print(f"🔎 Đang tìm các ID: {input_ids}...")
        for doc in documents:
            doc_id = get_id_from_url(doc['url'])
            if doc_id in input_ids:
                selected_docs.append(doc)
        
        if not selected_docs:
            print(f" Không tìm thấy tài liệu nào khớp với ID: {choice}")
        
        return selected_docs

    # 3. Tìm kiếm theo từ khóa (nếu nhập chữ)
    filtered = [doc for doc in documents if choice.lower() in doc['title'].lower()]
    if filtered:
        print(f"🔎 Tìm thấy {len(filtered)} tài liệu chứa từ khóa '{choice}':")
        for d in filtered:
            print(f"   - {d['title'][:80]}...")
        confirm = input("Bạn có muốn tải nhóm này không? (y/n): ").lower()
        if confirm == 'y':
            return filtered
            
    print(" Lựa chọn không hợp lệ hoặc không tìm thấy.")
    return []

def main():
    """Main function"""
    username = os.getenv("TDTU_USERNAME")
    password = os.getenv("TDTU_PASSWORD")
    
    if not username or not password:
        print(" Please set TDTU_USERNAME and TDTU_PASSWORD in .env file")
        return
    
    # Load documents list
    try:
        with open('tdtu_quy_che_list.json', 'r', encoding='utf-8') as f:
            documents = json.load(f)
    except FileNotFoundError:
        print(" tdtu_quy_che_list.json not found. Run crawl-quiche first.")
        return
    
    docs_to_process = filter_documents(documents)
    
    if not docs_to_process:
        print(" Đã hủy hoặc không có tài liệu để tải.")
        return
    # --------------------------------
    
    print(f"\n Đang chuẩn bị tải {len(docs_to_process)} tài liệu...\n")
    
    # Initialize converter
    converter = ImageToPDFConverter(output_dir="downloads_pdf")
    
    # Process documents
    results = []
    
    with TDTUClient(headless=False, verbose=True) as client:
        # Login
        print("🔐 Logging in...")
        if not client.login(username, password):
            print(" Login failed!")
            return
        
        print("✓ Login successful!\n")
        
        # Process each document
        for i, doc in enumerate(docs_to_process, 1):
            print(f"\n[{i}/{len(docs_to_process)}] {doc['title'][:60]}...")
            
            pdf_path = converter.process_document(client, doc)
            
            results.append({
                'title': doc['title'],
                'url': doc['url'],
                'pdf_path': pdf_path,
                'success': pdf_path is not None
            })
            
            if i < len(docs_to_process):
                time.sleep(2)
    
    print(f"\n{'='*70}")
    print("Summary:")
    print(f"{'='*70}")
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f" Processed: {len(results)} documents")
    print(f" Success:   {successful}")
    
    # Save results
    with open('download_pdf_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()