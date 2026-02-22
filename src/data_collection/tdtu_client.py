import os
import time
from typing import List, Dict
from loguru import logger
from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup


class TDTUClient:

    def __init__(self, headless: bool = False, verbose: bool = False):

        self.base_url = "https://stdportal.tdtu.edu.vn"
        self.quiche_base_url = "https://quychehocvu.tdtu.edu.vn"  # Base URL riêng cho Quy Chế
        self.quiche_url = "https://quychehocvu.tdtu.edu.vn/QuyChe"
        self.login_url = "https://stdportal.tdtu.edu.vn/Login/Index"
        self.verbose = verbose
        self.driver = None
        self.wait = None
        self.is_logged_in = False
        
        self.download_dir = os.path.join(os.getcwd(), "downloads_pdf")
        os.makedirs(self.download_dir, exist_ok=True)
        logger.info(f"Các file PDF sẽ được tải về tại: {self.download_dir}")

        # Setup Chrome options
        self.chrome_options = Options()
        if headless:
            self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--disable-gpu')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')
        self.chrome_options.add_argument('--window-size=1920,1080')
        self.chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        prefs = {
            "plugins.always_open_pdf_externally": True,  # Tắt trình xem PDF tích hợp
            "download.default_directory": self.download_dir, # Chỉ định thư mục tải
            "download.prompt_for_download": False, # Không hỏi trước khi tải
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True # Tắt các cảnh báo an toàn khi tải
        }
        self.chrome_options.add_experimental_option("prefs", prefs)

        logger.info("TDTU Client initialized")
    
    def init_driver(self):
        """Initialize Selenium WebDriver (Edge)"""
        if self.driver is None:
            try:
                # Use Edge instead of Chrome
                from selenium.webdriver.edge.options import Options as EdgeOptions
                
                edge_options = EdgeOptions()
                
                # Copy settings from chrome_options if headless
                if self.chrome_options and '--headless' in str(self.chrome_options.arguments):
                    edge_options.add_argument('--headless=new')
                    
                edge_options.add_argument('--no-sandbox')
                edge_options.add_argument('--disable-dev-shm-usage')
                edge_options.add_argument('--disable-gpu')
                edge_options.add_argument('--window-size=1920,1080')
                edge_options.add_argument('--disable-blink-features=AutomationControlled')
                edge_options.add_experimental_option('excludeSwitches', ['enable-logging', 'enable-automation'])
                edge_options.add_experimental_option('useAutomationExtension', False)
                
                # Use selenium-wire to capture network traffic
                seleniumwire_options = {
                    'disable_encoding': True,  # Ensure response body is readable
                    'verify_ssl': False,  # Disable SSL verification for proxy
                    'suppress_connection_errors': True  # Suppress connection errors
                }
                
                self.driver = webdriver.Edge(options=edge_options, seleniumwire_options=seleniumwire_options)
                self.wait = WebDriverWait(self.driver, 10)
                logger.info("Edge WebDriver with network capture initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize WebDriver: {e}")
                raise
    
    def login(self, username: str, password: str) -> bool:

        try:
            self.init_driver()
            
            logger.info(f"Attempting to login as {username}...")
            self.driver.get(self.login_url)
            time.sleep(3)
            
            # Find and fill login form - try multiple selectors
            try:
                username_input = None
                password_input = None
                
                # Try different selectors for username field
                username_selectors = [
                    (By.ID, "username"),
                    (By.ID, "Username"),
                    (By.NAME, "username"),
                    (By.NAME, "Username"),
                    (By.CSS_SELECTOR, "input[type='text']"),
                    (By.CSS_SELECTOR, "input[type='email']"),
                    (By.XPATH, "//input[@placeholder='Username' or @placeholder='Tên đăng nhập' or @placeholder='MSSV']")
                ]
                
                for by, selector in username_selectors:
                    try:
                        username_input = self.wait.until(
                            EC.presence_of_element_located((by, selector))
                        )
                        logger.info(f"Found username field using: {by}={selector}")
                        break
                    except TimeoutException:
                        continue
                
                if not username_input:
                    logger.error("Could not find username input field")
                    logger.error("Saved error page. Check tdtu_login_error_no_username.html for page structure")
                    return False
                
                # Try different selectors for password field
                password_selectors = [
                    (By.ID, "password"),
                    (By.ID, "Password"),
                    (By.NAME, "password"),
                    (By.NAME, "Password"),
                    (By.CSS_SELECTOR, "input[type='password']"),
                    (By.XPATH, "//input[@placeholder='Password' or @placeholder='Mật khẩu']")
                ]
                
                for by, selector in password_selectors:
                    try:
                        password_input = self.driver.find_element(by, selector)
                        logger.info(f"Found password field using: {by}={selector}")
                        break
                    except NoSuchElementException:
                        continue
                
                if not password_input:
                    logger.error("Could not find password input field")
                    return False
                
                # Clear and fill inputs
                username_input.clear()
                username_input.send_keys(username)
                logger.info("Username entered")
                
                password_input.clear()
                password_input.send_keys(password)
                logger.info("Password entered")
                
                if self.verbose:
                    logger.debug("Login form filled")
                
                # Find and click login button - try multiple selectors
                login_button = None
                button_selectors = [
                    (By.CSS_SELECTOR, "button[type='submit']"),
                    (By.XPATH, "//button[contains(text(), 'Login') or contains(text(), 'Đăng nhập') or contains(text(), 'ĐĂNG NHẬP')]"),
                    (By.CSS_SELECTOR, "input[type='submit']"),
                    (By.ID, "btnLogin"),
                    (By.ID, "loginButton"),
                    (By.CLASS_NAME, "btn-login"),
                    (By.XPATH, "//button[@type='submit']"),
                    (By.XPATH, "//input[@type='submit']")
                ]
                
                for by, selector in button_selectors:
                    try:
                        login_button = self.driver.find_element(by, selector)
                        logger.info(f"Found login button using: {by}={selector}")
                        break
                    except NoSuchElementException:
                        continue
                
                if not login_button:
                    logger.error("Could not find login button")
                    return False
                
                login_button.click()
                logger.info("Login button clicked")
                
                if self.verbose:
                    logger.debug("Waiting for redirect...")
                
                # Wait for redirect or error message
                time.sleep(4)
                
                # Check if login successful by checking URL or page content
                current_url = self.driver.current_url
                logger.info(f"Current URL after login attempt: {current_url}")

                
                if "Login" not in current_url and "login" not in current_url.lower():
                    self.is_logged_in = True
                    logger.info("Login successful!")
                    return True
                else:
                    # Check for error message
                    try:
                        error_selectors = [
                            (By.CLASS_NAME, "error-message"),
                            (By.CLASS_NAME, "alert-danger"),
                            (By.CLASS_NAME, "error"),
                            (By.XPATH, "//div[contains(@class, 'error') or contains(@class, 'alert')]")
                        ]
                        
                        error_msg = "Unknown error"
                        for by, selector in error_selectors:
                            try:
                                error_elem = self.driver.find_element(by, selector)
                                error_msg = error_elem.text
                                break
                            except NoSuchElementException:
                                continue
                        
                        logger.error(f"Login failed: {error_msg}")
                    except Exception:
                        logger.error("Login failed: Could not determine error message")

                    logger.error("Saved error page for debugging")
                    
                    self.is_logged_in = False
                    return False
                    
            except TimeoutException as e:
                logger.error(f"Login form not found - page timeout: {e}")
                logger.error("Check tdtu_login_timeout.html to see page structure")
                return False
            except NoSuchElementException as e:
                logger.error(f"Login form elements not found: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

        

    def navigate_to_quiche(self) -> bool:
        """Navigate to Quy Che page"""
        try:
            if not self.is_logged_in:
                logger.error("Not logged in. Please login first.")
                return False
            
            logger.info("Navigating to Quy Che page...")
            self.driver.get(self.quiche_url)
            time.sleep(3)
            
            # Check if page loaded successfully
            current_url = self.driver.current_url
            logger.info(f"Current URL: {current_url}")
            
            # Save page for debugging if verbose
            if self.verbose:
                logger.debug("Saved Quy Che page state")
            
            return True
            
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return False
    
    
    def extract_quiche_list(self) -> List[Dict[str, str]]:

        try:
            if not self.is_logged_in:
                logger.error("Not logged in")
                return []
            
            # Navigate to Quy Che page
            if not self.navigate_to_quiche():
                return []
            
            logger.info("Extracting Quy Che documents from all pages...")
            
            # Wait for dynamic content to load (DataTables)
            logger.info("Waiting for dynamic content to load...")
            time.sleep(5)
            
            all_documents = []
            current_page = 1
            
            while True:
                logger.info(f"📄 Processing page {current_page}...")
                
                # Parse current page with BeautifulSoup
                soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                
                # Find list items (the actual documents)
                list_items = soup.find_all('div', class_='list-item')
                logger.info(f"  Found {len(list_items)} list items on page {current_page}")
                
                # Extract documents from current page
                page_docs = []
                for item in list_items:
                    try:
                        # Find the link and title
                        link = item.find('a')
                        if not link:
                            continue
                        
                        # Get title (full title from 'title' attribute, or text)
                        title = link.get('title', '') or link.get_text(strip=True)
                        
                        # Get URL
                        url = link.get('href', '')
                        if url and not url.startswith('http'):
                            # Sử dụng quiche_base_url cho các link Quy Chế
                            url = f"{self.quiche_base_url}{url}" if url.startswith('/') else f"{self.quiche_base_url}/{url}"
                        
                        # Get metadata (department and date)
                        metadata_span = item.find('span', style=lambda value: value and 'float: right' in value)
                        department = ''
                        date = ''
                        
                        if metadata_span:
                            metadata_text = metadata_span.get_text(strip=True)
                            # Format: "Department | Date"
                            parts = metadata_text.split('|')
                            if len(parts) >= 2:
                                department = parts[0].strip()
                                date = parts[1].strip()
                            elif len(parts) == 1:
                                date = parts[0].strip()
                        
                        # Only add if we have a meaningful title
                        if title and len(title) > 5:
                            doc_info = {
                                'title': title,
                                'url': url,
                                'type': 'quy_che',
                                'issue_date': date,
                                'department': department,
                                'status': 'Active',  # Assume active if it's listed
                                'page': current_page
                            }
                            page_docs.append(doc_info)
                            
                            if self.verbose:
                                logger.debug(f"  Found document: {title[:50]}... | {department} | {date}")
                    
                    except Exception as e:
                        logger.warning(f"Error parsing list item: {e}")
                        continue
                
                all_documents.extend(page_docs)
                logger.info(f"  ✓ Extracted {len(page_docs)} documents from page {current_page}")
                
                # Check if we've reached page 3 (based on screenshot showing 3 pages)
                if current_page >= 3:
                    logger.info(f"  Reached last page (page 3)")
                    break
                
                # Check if there's a next page button
                try:
                    # Find pagination buttons - look for next page number
                    next_page_num = current_page + 1
                    
                    # Try to find and click the next page button
                    # Common pagination patterns: page number buttons, "Next" button
                    next_button = None
                    
                    # Try finding by page number in Bootstrap pagination
                    try:
                        next_button = self.driver.find_element("xpath", f"//li[@class='page-item']//a[text()='{next_page_num}']")
                    except:
                        pass
                    
                    # Try finding by data attribute
                    if not next_button:
                        try:
                            next_button = self.driver.find_element("xpath", f"//a[@data-page='{next_page_num}']")
                        except:
                            pass
                    
                    # Try finding by direct page number
                    if not next_button:
                        try:
                            next_button = self.driver.find_element("xpath", f"//a[contains(@class, 'page') and text()='{next_page_num}']")
                        except:
                            pass
                    
                    if next_button:
                        # Check if button is disabled/inactive
                        button_classes = next_button.get_attribute('class') or ''
                        parent_classes = ''
                        try:
                            parent = next_button.find_element("xpath", "..")
                            parent_classes = parent.get_attribute('class') or ''
                        except:
                            pass
                        
                        if 'disabled' in button_classes or 'disabled' in parent_classes or 'active' in button_classes:
                            logger.info(f"  No more pages (next button is disabled/active)")
                            break
                        
                        logger.info(f"  → Clicking to page {next_page_num}...")
                        next_button.click()
                        time.sleep(3)  # Wait for page to load
                        current_page += 1
                    else:
                        logger.info(f"  No more pages (next button not found)")
                        break
                        
                except Exception as e:
                    logger.info(f"  No more pages: {e}")
                    break
            
            logger.info(f"Found total {len(all_documents)} quy che documents across {current_page} page(s)")
            
            # Save HTML for inspection if no documents found
            if not all_documents:
                logger.warning("No documents found. Saving page for inspection...")
            
            return all_documents
            
        except Exception as e:
            logger.error(f"Error extracting quy che list: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    

    
    def close(self):
        """Close browser and cleanup"""
        if self.driver:
            logger.info("Closing browser...")
            self.driver.quit()
            self.driver = None
            self.wait = None
            self.is_logged_in = False
    
    def __enter__(self):
        """Context manager entry"""
        self.init_driver()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


if __name__ == "__main__":
    # Test the client
    from dotenv import load_dotenv
    load_dotenv()
    
    username = os.getenv("TDTU_USERNAME")
    password = os.getenv("TDTU_PASSWORD")
    
    if not username or not password:
        print("Vui lòng thiết lập TDTU_USERNAME và TDTU_PASSWORD trong file .env")
        exit(1)
    
    # Sử dụng context manager
    with TDTUClient(headless=False, verbose=True) as client:
        # Đăng nhập
        if client.login(username, password):
            print("✓ Đăng nhập thành công!")
            
            # Trích xuất danh sách quy chế
            quy_che_list = client.extract_quiche_list()
            if quy_che_list:
                print(f"\n Tìm thấy {len(quy_che_list)} tài liệu")
                for i, doc in enumerate(quy_che_list[:5], 1):  # Hiển thị 5 cái đầu
                    print(f"{i}. {doc.get('title', 'N/A')}")

            else:
                print(" Không tìm thấy tài liệu nào.")
            
            input("\nNhấn Enter để đóng trình duyệt...")
        else:
            print("✗ Đăng nhập thất bại!")