import csv
import pandas as pd
import MetaTrader5 as mt5
import time
import os
from datetime import datetime
import threading
import logging
from flask import Flask, request, jsonify
import json
import urllib.parse
import argparse
import sys
import queue
from collections import deque
from typing import Dict, Set, Deque, Optional
import socket
from concurrent.futures import ThreadPoolExecutor
import psutil  # Add at top with other imports
import subprocess
import atexit
import random
import signal

# Setup advanced logging
logger = logging.getLogger(__name__)

# Constants for ultra-reliable execution
MAX_RETRIES = 5
RETRY_DELAY = 0.1  # 100ms between retries
MARKET_CHECK_INTERVAL = 0.5  # Check market status every 500ms
CONNECTION_CHECK_INTERVAL = 0.1 # Check connection every 1 second
MAX_QUEUE_SIZE = 1000
EXECUTOR_THREADS = 4

# Add these constants near other constants
MEMORY_CHECK_INTERVAL = 5  # Check every 5 seconds
MAX_MEMORY_PERCENT = 85  # Maximum memory percentage before action
MONITORED_PROCESSES = ["UiRobot.exe"]  # List of processes to monitor
MEMORY_WARNING_THRESHOLD = 75  # Warning threshold percentage
UIROBOT_CHECK_INTERVAL = 5  # Check UiRobot every 5 seconds
UIROBOT_INITIAL_WAIT = 90  # Wait 2 minutes before capturing command line

# Port configuration
PORT = 3000  # We will only use port 3000
PORT_LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'port_3000.lock')

# Single instance lock file
LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eibmemory.lock')
PORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eibmemory.port')

# Add these constants
CMDLINE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uirobot_cmdline.txt')

def find_free_port():
    """Find a free port in the range 3000-3999"""
    try:
        # First try the default port
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', PORT))
        sock.close()
        
        if result != 0:  # Port is free
            return PORT
            
        # Try random ports in range if default is taken
        used_ports = set()
        while len(used_ports) < (PORT_RANGE_END - PORT_RANGE_START):
            port = random.randint(PORT_RANGE_START, PORT_RANGE_END)
            if port in used_ports:
                continue
                
            used_ports.add(port)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result != 0:  # Port is free
                return port
                
        raise Exception("No free ports found in range")
        
    except Exception as e:
        logger.error(f"Error finding free port: {e}")
        return None

def save_port_number(port):
    """Save the current port number to file"""
    try:
        with open(PORT_FILE, 'w') as f:
            f.write(str(port))
    except Exception as e:
        logger.error(f"Error saving port number: {e}")

def get_saved_port():
    """Get the saved port number"""
    try:
        if os.path.exists(PORT_FILE):
            with open(PORT_FILE, 'r') as f:
                return int(f.read().strip())
    except:
        pass
    return None

def cleanup_port_file():
    """Remove port file on exit"""
    try:
        if os.path.exists(PORT_FILE):
            os.remove(PORT_FILE)
    except:
        pass

def ensure_single_instance():
    """Ensure only one instance of the script runs at a time"""
    try:
        if os.path.exists(LOCK_FILE):
            # Check if process is still running
            with open(LOCK_FILE, 'r') as f:
                old_pid = int(f.read().strip())
                try:
                    # Check if process exists
                    process = psutil.Process(old_pid)
                    if process.name().endswith('.py') or process.name().endswith('.exe'):
                        logger.error(f"Another instance is already running (PID: {old_pid})")
                        sys.exit(1)
                except psutil.NoSuchProcess:
                    # Process doesn't exist, we can continue
                    pass
        
        # Write our PID to lock file
        with open(LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
            
        # Register cleanup on exit
        atexit.register(cleanup_lock_file)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in single instance check: {e}")
        sys.exit(1)

def cleanup_lock_file():
    """Remove lock file on exit"""
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except:
        pass

def save_cmdline(cmdline):
    """Save UiRobot command line arguments to file"""
    try:
        with open(CMDLINE_FILE, 'w') as f:
            if cmdline:
                f.write('\n'.join(cmdline))
    except Exception as e:
        logger.error(f"Error saving command line: {e}")

def load_cmdline():
    """Load saved UiRobot command line arguments"""
    try:
        if os.path.exists(CMDLINE_FILE):
            with open(CMDLINE_FILE, 'r') as f:
                lines = f.read().splitlines()
                if lines:
                    return lines
    except Exception as e:
        logger.error(f"Error loading command line: {e}")
    return None

def get_uirobot_cmdline():
    """Get UiRobot.exe command line arguments - only once"""
    # First try to load saved command line
    saved_cmdline = load_cmdline()
    if saved_cmdline:
        logger.info("Using saved UiRobot command line arguments")
        return saved_cmdline
        
    try:
        for proc in psutil.process_iter(['name', 'cmdline']):
            if proc.info['name'] == 'UiRobot.exe':
                cmdline = proc.info['cmdline']
                if cmdline and len(cmdline) > 1:
                    # Process command line to ensure port 3000
                    new_args = []
                    for arg in cmdline[1:]:
                        # Replace any port number with 3000
                        if 'localhost:' in arg or '127.0.0.1:' in arg:
                            parts = arg.split(':')
                            if len(parts) > 1:
                                # Always use port 3000
                                parts[-1] = str(PORT)
                                arg = ':'.join(parts)
                        new_args.append(arg)
                    
                    # Save the command line for future use
                    save_cmdline(new_args)
                    logger.info(f"Captured and saved UiRobot command line: {new_args}")
                    return new_args
        return []
    except Exception as e:
        logger.error(f"Error getting UiRobot command line: {e}")
        return []

def kill_uipath_processes():
    """Kill all UiPath related processes"""
    try:
        processes_to_kill = ["UiRobot.exe", "UiPath.Executor.exe"]
        killed = []
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['name'] in processes_to_kill:
                    proc.kill()
                    proc.wait(timeout=5)
                    killed.append(proc.info['name'])
                    logger.info(f"Killed process: {proc.info['name']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                continue
                
        return killed
    except Exception as e:
        logger.error(f"Error killing UiPath processes: {e}")
        return []

class UiRobotMonitor:
    def __init__(self):
        self.monitoring = True
        self.last_restart = 0
        self.restart_count = 0
        self.captured_cmdline = None
        self.initial_wait_complete = False
        self.start_time = time.time()
        
    def is_uirobot_running(self):
        """Check if UiRobot is running"""
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] == 'UiRobot.exe':
                    return True
            return False
        except:
            return False  # Assume not running if we can't check
        
    def capture_cmdline(self):
        """Capture UiRobot command line after initial wait - only once"""
        try:
            if not self.initial_wait_complete:
                elapsed = time.time() - self.start_time
                if elapsed >= UIROBOT_INITIAL_WAIT:
                    logger.info("Initial wait complete - capturing UiRobot command line")
                    self.captured_cmdline = get_uirobot_cmdline()
                    if self.captured_cmdline:
                        logger.info(f"Using command line arguments: {self.captured_cmdline}")
                    else:
                        # Try to load saved command line as fallback
                        self.captured_cmdline = load_cmdline()
                        if self.captured_cmdline:
                            logger.info(f"Using previously saved command line: {self.captured_cmdline}")
                        else:
                            logger.warning("No command line arguments found for UiRobot")
                    self.initial_wait_complete = True
        except Exception as e:
            logger.error(f"Error capturing command line: {e}")
            # Continue even if capture fails
                
    def restart_uirobot(self):
        """Restart UiRobot.exe with captured command line - NEVER GIVE UP"""
        while True:  # Keep trying forever
            try:
                # Find UiRobot.exe path
                uirobot_paths = [
                    r"C:\Program Files\UiPath\Studio\UiRobot.exe",
                    r"C:\Program Files (x86)\UiPath\Studio\UiRobot.exe"
                ]
                
                uirobot_path = None
                for path in uirobot_paths:
                    if os.path.exists(path):
                        uirobot_path = path
                        break
                        
                if not uirobot_path:
                    logger.error("UiRobot.exe not found - retrying in 5 seconds...")
                    time.sleep(5)
                    continue  # Keep trying
                    
                # Kill all UiPath processes including UiPath.Executor.exe
                retry_count = 0
                while retry_count < 3:  # Try killing processes up to 3 times
                    logger.info("Terminating all UiPath processes...")
                    killed_processes = kill_uipath_processes()
                    if killed_processes:
                        logger.info(f"Successfully terminated: {', '.join(killed_processes)}")
                        time.sleep(2)  # Give processes time to fully terminate
                        break
                    retry_count += 1
                    time.sleep(1)
                
                # Use captured command line if available, otherwise try to load saved
                cmdline = self.captured_cmdline
                if not cmdline:
                    cmdline = load_cmdline()
                if not cmdline:
                    cmdline = []  # Use empty args if none found
                
                # Start new UiRobot process
                subprocess.Popen([uirobot_path] + cmdline)
                logger.info(f"Started UiRobot.exe with args: {cmdline}")
                
                # Verify process started
                time.sleep(5)  # Wait for startup
                if self.is_uirobot_running():
                    logger.info("UiRobot.exe successfully started and running")
                    return True
                else:
                    logger.error("UiRobot.exe failed to start - retrying...")
                    continue  # Keep trying
                
            except Exception as e:
                logger.error(f"Error restarting UiRobot: {e}")
                logger.info("Retrying restart in 5 seconds...")
                time.sleep(5)
                continue  # Never give up, keep trying
            
    def monitor_uirobot(self):
        """Monitor UiRobot and restart if needed - RUNS FOREVER"""
        logger.info("Starting UNSTOPPABLE UiRobot monitoring...")
        
        while True:  # Changed to True - monitor forever
            try:
                # First, try to capture command line if not done yet
                self.capture_cmdline()
                
                if not self.is_uirobot_running():
                    current_time = time.time()
                    
                    # Only attempt restart after initial wait
                    if not self.initial_wait_complete:
                        logger.info("Waiting for initial period before monitoring UiRobot")
                        time.sleep(UIROBOT_CHECK_INTERVAL)
                        continue
                    
                    # Reset restart count every hour
                    if current_time - self.last_restart > 3600:
                        self.restart_count = 0
                    
                    # Always try to restart - no maximum limit
                    logger.warning(f"UiRobot.exe not running - attempting restart #{self.restart_count + 1}")
                    if self.restart_uirobot():
                        self.last_restart = current_time
                        self.restart_count += 1
                        logger.info(f"UiRobot.exe restarted (total restarts: {self.restart_count})")
                    
            except Exception as e:
                logger.error(f"Error in UiRobot monitor: {e}")
                logger.info("Continuing monitoring despite error...")
                
            finally:
                # Always keep monitoring
                try:
                    time.sleep(UIROBOT_CHECK_INTERVAL)
                except:
                    pass  # Ignore sleep interrupts
                continue  # Ensure we keep going no matter what

# Global trade queue for ultra-fast processing
trade_queue: queue.Queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
pending_trades: Deque = deque(maxlen=MAX_QUEUE_SIZE)
executed_trades: Dict[str, Dict] = {}
failed_trades: Set[str] = set()

# Thread pool for parallel trade processing
trade_executor = ThreadPoolExecutor(max_workers=EXECUTOR_THREADS)

# Connection monitoring
class ConnectionMonitor:
    def __init__(self):
        self.last_check = time.time()
        self.is_connected = False
        self.reconnect_count = 0
        self.max_reconnects = 10
        
    def check_internet(self) -> bool:
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=1)
            return True
        except OSError:
            return False
    
    def check_mt5_connection(self) -> bool:
        try:
            if not mt5.initialize():
                logger.error("MT5 initialization failed")
                return False
            return mt5.account_info() is not None
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
            
    def ensure_connection(self) -> bool:
        current_time = time.time()
        if current_time - self.last_check < CONNECTION_CHECK_INTERVAL:
            return self.is_connected
            
        self.last_check = current_time
        
        if not self.check_internet():
            logger.error("No internet connection")
            self.is_connected = False
            return False
            
        if not self.check_mt5_connection():
            logger.error("MT5 connection lost")
            self.is_connected = False
            
            while self.reconnect_count < self.max_reconnects:
                logger.info(f"Attempting to reconnect to MT5 (attempt {self.reconnect_count + 1})")
                if mt5.initialize():
                    logger.info("Successfully reconnected to MT5")
                    self.is_connected = True
                    self.reconnect_count = 0
                    return True
                    
                self.reconnect_count += 1
                time.sleep(1)
                
            logger.critical("Failed to reconnect to MT5 after maximum attempts")
            return False
            
        self.is_connected = True
        return True

# Global connection monitor
connection_monitor = ConnectionMonitor()

# Trade processing status tracking
class TradeStatus:
    PENDING = "PENDING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRYING = "RETRYING"

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='MT5 Bitcoin Trading System with Flask API')
    parser.add_argument('--lot-percentage', type=float, default=100.0,
                       help='Percentage of lot amount to execute (1-100, default: 100)')
    args = parser.parse_args()
    
    # Validate lot percentage
    if args.lot_percentage < 1 or args.lot_percentage > 100:
        print(f"Error: lot-percentage must be between 1 and 100, got {args.lot_percentage}")
        sys.exit(1)
    
    return args

# CONFIGURATION: Always trade on Bitcoin
TRADING_SYMBOL = "XAUUSD"  # Correct Bitcoin symbol (was incorrectly set to XAUUSD which is Gold)

# Global variable to store received trade data
received_trade_data = []
data_lock = threading.Lock()
trade_update_event = threading.Event()  # Event to signal new trade data

# Flask app setup
app = Flask(__name__)

# Configure Flask logging
app.logger.setLevel(logging.INFO)

@app.route('/trades', methods=['POST'])
def receive_trades():
    """Receive trade data via POST request - handles both JSON and plain text - ALWAYS returns 200"""
    global received_trade_data
    
    logger.info("Received POST request to /trades")
    
    data = None
    error_messages = []
    
    # Try to get data from request
    try:
        if request.is_json:
            data = request.get_json()
            logger.info(f"Received JSON data: {data}")
        else:
            # Handle plain text data
            text_data = request.get_data(as_text=True)
            logger.info(f"Received text data: {text_data}")
            
            if not text_data.strip():
                logger.info("Empty text data received - will close all trades")
                data = []  # Empty data means close all trades
            else:
                # Parse text data
                data = parse_text_data(text_data)
                logger.info(f"Parsed text to data: {data}")
                
    except Exception as e:
        error_msg = f"Error parsing request data: {str(e)}"
        logger.warning(error_msg)
        error_messages.append(error_msg)
        data = []  # Default to empty data
    
    # If no data was parsed, that's okay - it means close all trades
    if data is None:
        logger.info("No data received - treating as close all trades command")
        data = []
    
    # Parse the received data
    trades = []
    
    try:
        # Handle both single trade and array of trades
        if isinstance(data, list):
            trade_list = data
        else:
            trade_list = [data] if data else []
        
        logger.info(f"Processing {len(trade_list)} trade items")
        
        for i, trade_data in enumerate(trade_list):
            try:
                logger.info(f"Processing trade item {i+1}: {trade_data}")
                
                # Extract trade information from received data
                trade_info = {
                    'REF': str(trade_data.get('REF', '')),
                    'Trade_Type': str(trade_data.get('Trade_Type', '')),
                    'Bought_Lot': float(trade_data.get('Bought_Lot', 0.0)),
                    'Sold_Lot': float(trade_data.get('Sold_Lot', 0.0)),
                    'Symbol': str(trade_data.get('Symbol', '')),
                    'Timestamp': str(trade_data.get('Timestamp', ''))
                }
                
                logger.info(f"Parsed trade info: {trade_info}")
                
                # Validate required fields
                if trade_info['REF'] and trade_info['Trade_Type']:
                    trades.append(trade_info)
                    logger.info(f"Valid trade added: REF={trade_info['REF']}")
                else:
                    warning_msg = f"Incomplete trade data for item {i+1}: missing REF or Trade_Type"
                    logger.warning(warning_msg)
                    error_messages.append(warning_msg)
                    
            except (ValueError, KeyError, TypeError) as e:
                error_msg = f"Error processing trade data item {i+1}: {str(e)}"
                logger.warning(error_msg)
                error_messages.append(error_msg)
                continue
    
    except Exception as e:
        error_msg = f"Error processing trade list: {str(e)}"
        logger.warning(error_msg)
        error_messages.append(error_msg)
    
    # Update global trade data with thread safety
    try:
        logger.info(f"Updating global trade data with {len(trades)} trades")
        with data_lock:
            received_trade_data = trades
        
        trade_update_event.set()  # Signal that new data is available
        logger.info(f"Successfully received {len(trades)} trades via POST request")
        for trade in trades:
            logger.info(f"  REF: {trade['REF']}, Type: {trade['Trade_Type']}, Symbol: {trade['Symbol']}")
        
    except Exception as e:
        error_msg = f"Error updating trade data: {str(e)}"
        logger.warning(error_msg)
        error_messages.append(error_msg)
    
    # Prepare response - ALWAYS 200 status
    if len(trades) > 0:
        message = f"Successfully received {len(trades)} trades"
        if error_messages:
            message += f" (with {len(error_messages)} warnings)"
    elif len(error_messages) > 0:
        message = f"Request processed with {len(error_messages)} issues - no valid trades found"
    else:
        message = "Request processed - no trades to execute (close all trades mode)"
    
    response_data = {
        "status": "success",  # Always success
        "message": message,
        "trades_count": len(trades),
        "warnings": error_messages if error_messages else None
    }
    
    logger.info(f"Sending response (200): {response_data}")
    
    # ALWAYS return 200 status code
    return jsonify(response_data), 200

def parse_text_data(text_data):
    """Parse plain text data into trade format - handles URL-encoded form data and JSON"""
    try:
        logger.info(f"=== PARSING TEXT DATA ===")
        logger.info(f"Raw text length: {len(text_data)}")
        logger.info(f"Raw text: '{text_data[:200]}...'")  # Show first 200 chars
        
        if not text_data or not text_data.strip():
            logger.error("Empty or whitespace-only text data")
            return []
        
        # Clean the text data
        text_data = text_data.strip()
        
        # Method 1: Check if it's URL-encoded form data (like "a=%5B%7B...")
        try:
            # Check if it starts with a form parameter pattern
            if '=' in text_data and '%' in text_data:
                logger.info("Detected URL-encoded form data")
                
                # Parse as form data
                parsed_data = urllib.parse.parse_qs(text_data)
                logger.info(f"Parsed form data keys: {list(parsed_data.keys())}")
                
                # Look for JSON in any of the form parameters
                json_data = None
                for key, values in parsed_data.items():
                    if values:
                        # URL decode the value
                        decoded_value = urllib.parse.unquote(values[0])
                        logger.info(f"Decoded value for key '{key}': {decoded_value[:200]}...")
                        
                        # Try to parse as JSON
                        try:
                            import json
                            json_data = json.loads(decoded_value)
                            logger.info(f"Successfully parsed JSON from form parameter '{key}'")
                            break
                        except json.JSONDecodeError:
                            logger.info(f"Value for key '{key}' is not valid JSON")
                            continue
                
                if json_data:
                    return parse_json_trade_data(json_data)
                    
        except Exception as e:
            logger.warning(f"URL-encoded form parsing failed: {e}")
        
        # Method 2: Try direct URL decoding
        try:
            decoded_text = urllib.parse.unquote(text_data)
            logger.info(f"URL decoded text: {decoded_text[:200]}...")
            
            # Try to parse as JSON
            try:
                import json
                json_data = json.loads(decoded_text)
                logger.info("Successfully parsed URL-decoded text as JSON")
                return parse_json_trade_data(json_data)
            except json.JSONDecodeError:
                logger.info("URL-decoded text is not valid JSON")
        except Exception as e:
            logger.warning(f"URL decoding failed: {e}")
        
        # Method 3: Try to parse as direct JSON
        try:
            import json
            json_data = json.loads(text_data)
            logger.info("Successfully parsed text as direct JSON")
            return parse_json_trade_data(json_data)
        except json.JSONDecodeError:
            logger.info("Text is not valid JSON")
        
        # Method 4: Fall back to original CSV/text parsing
        logger.info("Falling back to CSV/text parsing methods")
        return parse_csv_text_data(text_data)
        
    except Exception as e:
        logger.error(f"Fatal error in parse_text_data: {e}")
        logger.exception("Full traceback:")
        return []

def parse_json_trade_data(json_data):
    """Parse JSON trade data into the expected format"""
    try:
        logger.info(f"=== PARSING JSON TRADE DATA ===")
        logger.info(f"JSON data type: {type(json_data)}")
        
        if isinstance(json_data, list):
            trade_list = json_data
        else:
            trade_list = [json_data]
        
        logger.info(f"Processing {len(trade_list)} JSON trade items")
        
        trades = []
        for i, item in enumerate(trade_list, 1):
            try:
                logger.info(f"Processing JSON item {i}: {item}")
                
                # Extract trade information - handle both old and new JSON formats
                ref = item.get('Ref', '')
                
                # Handle both "Sell/Buy" (old format) and "B/S" (new format)
                sell_buy = item.get('Sell/Buy', '') or item.get('B/S', '')
                
                lot = item.get('Lot', '')
                bought_lot = item.get('Bought Lot', '')
                sold_lot = item.get('Sold Lot', '')
                contract = item.get('Contract', '')
                
                # Determine trade type
                trade_type = ''
                if sell_buy == 'B':
                    trade_type = 'Buy'
                elif sell_buy == 'S':
                    trade_type = 'Sell'
                
                # Determine lot amounts
                bought_lot_val = 0.0
                sold_lot_val = 0.0
                
                if trade_type == 'Buy':
                    if bought_lot:
                        try:
                            bought_lot_val = float(bought_lot)
                        except ValueError:
                            pass
                    elif lot:
                        try:
                            bought_lot_val = float(lot)
                        except ValueError:
                            pass
                elif trade_type == 'Sell':
                    if sold_lot:
                        try:
                            sold_lot_val = float(sold_lot)
                        except ValueError:
                            pass
                    elif lot:
                        try:
                            sold_lot_val = float(lot)
                        except ValueError:
                            pass
                
                # Skip invalid entries (like summary rows with Liquidation: "-1" or empty Ref)
                liquidation = item.get('Liquidation', '')
                if liquidation == '-1' or not ref or not trade_type or ref == '1':
                    logger.info(f"Skipping invalid/summary item {i}: Liquidation={liquidation}, Ref={ref}, Type={trade_type}")
                    continue
                
                trade_data = {
                    'REF': ref,
                    'Trade_Type': trade_type,
                    'Bought_Lot': bought_lot_val,
                    'Sold_Lot': sold_lot_val,
                    'Symbol': contract or 'UNKNOWN',
                    'Timestamp': item.get('Tran. Date', '')
                }
                
                logger.info(f"Parsed JSON trade {i}: {trade_data}")
                trades.append(trade_data)
                
            except Exception as e:
                logger.warning(f"Error processing JSON item {i}: {e}")
                continue
        
        logger.info(f"=== JSON PARSING COMPLETE ===")
        logger.info(f"Successfully parsed {len(trades)} trades from JSON")
        return trades
        
    except Exception as e:
        logger.error(f"Error in parse_json_trade_data: {e}")
        return []

def parse_csv_text_data(text_data):
    """Original CSV/text parsing methods"""
    try:
        logger.info("Using original CSV/text parsing methods")
        
        lines = text_data.split('\n')
        trades = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            logger.info(f"Processing line {line_num}: '{line[:100]}...'")
            
            # Try multiple parsing methods
            trade_data = None
            
            # Method 1: Try comma-separated values
            try:
                fields = [f.strip() for f in line.split(',')]
                if len(fields) >= 2:
                    trade_data = {
                        'REF': fields[0],
                        'Trade_Type': fields[1],
                        'Bought_Lot': float(fields[2]) if len(fields) > 2 and fields[2] else 0.0,
                        'Sold_Lot': float(fields[3]) if len(fields) > 3 and fields[3] else 0.0,
                        'Symbol': fields[4] if len(fields) > 4 else 'UNKNOWN',
                        'Timestamp': fields[5] if len(fields) > 5 else ''
                    }
            except (ValueError, IndexError):
                pass
            
            # Method 2: Try space-separated values
            if not trade_data:
                try:
                    fields = line.split()
                    if len(fields) >= 2:
                        trade_data = {
                            'REF': fields[0],
                            'Trade_Type': fields[1],
                            'Bought_Lot': float(fields[2]) if len(fields) > 2 else 0.0,
                            'Sold_Lot': float(fields[3]) if len(fields) > 3 else 0.0,
                            'Symbol': fields[4] if len(fields) > 4 else 'UNKNOWN',
                            'Timestamp': ' '.join(fields[5:]) if len(fields) > 5 else ''
                        }
                except (ValueError, IndexError):
                    pass
            
            # Method 3: Try simple REF-only
            if not trade_data:
                if line.replace(' ', '').replace('-', '').replace('_', '').isalnum():
                    trade_data = {
                        'REF': line,
                        'Trade_Type': 'Buy',
                        'Bought_Lot': 1.0,
                        'Sold_Lot': 0.0,
                        'Symbol': 'UNKNOWN',
                        'Timestamp': ''
                    }
            
            if trade_data and trade_data.get('REF') and trade_data.get('Trade_Type'):
                trades.append(trade_data)
                logger.info(f"✅ Successfully added trade from line {line_num}: REF={trade_data['REF']}")
        
        return trades
        
    except Exception as e:
        logger.error(f"Error in parse_csv_text_data: {e}")
        return []

@app.route('/status', methods=['GET'])
def get_status():
    """Get current system status - ALWAYS returns 200"""
    logger.info("Received GET request to /status")
    
    try:
        global received_trade_data
        
        with data_lock:
            trade_count = len(received_trade_data)
        
        status_data = {
            "status": "running",
            "trades_count": trade_count,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Status response: {status_data}")
        return jsonify(status_data), 200
        
    except Exception as e:
        error_msg = f"Error getting status: {str(e)}"
        logger.warning(error_msg)
        
        # Even if there's an error, return 200 with error info
        error_status = {
            "status": "error", 
            "message": error_msg,
            "trades_count": 0,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(error_status), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()}), 200

def start_flask_server():
    """Start Flask server in a separate thread"""
    logger.info(f"Starting Flask server on localhost:{PORT}...")
    try:
        # Ensure port is available
        if not ensure_port_available():
            logger.error("Could not secure port 3000")
            sys.exit(1)
            
        # Start port monitor
        port_monitor = PortMonitor()
        port_monitor.start()
        
        # Enable threaded mode for better concurrent handling
        app.run(host='localhost', port=PORT, debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        logger.error(f"Error starting Flask server: {e}")
        logger.exception("Full traceback:")

class MT5TradeManager:
    def __init__(self, lot_percentage):
        self.active_trades = {}  # Dictionary to store active trades by REF
        self.manually_closed_refs = set()  # Track REFs that were manually closed
        self.monitoring = True
        self._lot_percentage = lot_percentage
        self.mt5_connected = False
        self.last_data_check = datetime.now()
        self.trade_status = {}  # Track status of each trade
        self.retry_counts = {}  # Track retry attempts for each trade
        self.last_market_check = 0
        self.market_is_open = False
        self._lot_lock = threading.Lock()  # Lock for thread-safe lot updates
        
    @property
    def lot_percentage(self):
        with self._lot_lock:
            return self._lot_percentage
            
    @lot_percentage.setter
    def lot_percentage(self, value):
        """Thread-safe way to update lot percentage without restart"""
        if not 0 < value <= 100:
            logger.error(f"Invalid lot percentage {value}. Must be between 0 and 100")
            return
            
        with self._lot_lock:
            self._lot_percentage = value / 100.0
            logger.info(f"Lot percentage updated to {value}%")
            
    def update_lot_size(self, new_percentage: float) -> bool:
        """Update lot size dynamically without requiring restart"""
        try:
            if not 0 < new_percentage <= 100:
                logger.error(f"Invalid lot percentage {new_percentage}. Must be between 0 and 100")
                return False
                
            self.lot_percentage = new_percentage
            logger.info(f"Successfully updated lot size to {new_percentage}%")
            return True
            
        except Exception as e:
            logger.error(f"Error updating lot size: {e}")
            return False
    
    def is_market_open(self) -> bool:
        """Check if market is open with caching to prevent excessive checks"""
        current_time = time.time()
        if current_time - self.last_market_check < MARKET_CHECK_INTERVAL:
            return self.market_is_open
            
        self.last_market_check = current_time
        tick = mt5.symbol_info_tick(TRADING_SYMBOL)
        self.market_is_open = tick is not None
        return self.market_is_open
    
    def execute_trade_with_retry(self, trade_info: dict) -> Optional[dict]:
        """Execute trade with advanced retry logic and error handling"""
        ref = trade_info['REF']
        self.trade_status[ref] = TradeStatus.PENDING
        self.retry_counts[ref] = 0
        
        while self.retry_counts[ref] < MAX_RETRIES:
            try:
                # Ensure connection before each attempt
                if not connection_monitor.ensure_connection():
                    logger.error(f"No connection available for trade {ref}")
                    time.sleep(RETRY_DELAY)
                    self.retry_counts[ref] += 1
                    continue
                
                # Check if market is open
                if not self.is_market_open():
                    logger.warning(f"Market closed for trade {ref}")
                    return None
                
                # Check for duplicate before each attempt
                if self.is_duplicate_trade(ref):
                    logger.info(f"Duplicate trade detected for {ref}")
                    return None
                
                self.trade_status[ref] = TradeStatus.EXECUTING
                
                # Get symbol info and validate
                symbol_info = mt5.symbol_info(TRADING_SYMBOL)
                if not symbol_info or not symbol_info.visible:
                    if not mt5.symbol_select(TRADING_SYMBOL, True):
                        raise Exception(f"Failed to select symbol {TRADING_SYMBOL}")
                
                # Calculate and validate lot size
                trade_type = trade_info['Trade_Type']
                original_lot = trade_info['Bought_Lot'] if trade_type == 'Buy' else trade_info['Sold_Lot']
                lot_size = original_lot * self.lot_percentage
                
                # Ensure lot size meets symbol requirements
                lot_size = max(symbol_info.volume_min, min(symbol_info.volume_max, lot_size))
                lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
                
                # Get current price with slippage protection
                tick = mt5.symbol_info_tick(TRADING_SYMBOL)
                if not tick:
                    raise Exception("Failed to get current price")
                
                price = tick.ask if trade_type == 'Buy' else tick.bid
                
                # Prepare order with FOK filling
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": TRADING_SYMBOL,
                    "volume": lot_size,
                    "type": mt5.ORDER_TYPE_BUY if trade_type == 'Buy' else mt5.ORDER_TYPE_SELL,
                    "price": price,
                    "deviation": 10,  # Tight deviation for accurate copying
                    "magic": int(ref),
                    "comment": f"REF{ref}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                
                # Execute trade
                result = mt5.order_send(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Success! Record the trade with actual execution details
                    trade_record = {
                        'ref': ref,
                        'ticket': result.order,
                        'symbol': TRADING_SYMBOL,
                        'original_symbol': trade_info['Symbol'],
                        'type': trade_type,
                        'volume': result.volume,
                        'open_price': result.price,
                        'magic': int(ref),
                        'execution_time': datetime.now().timestamp()
                    }
                    
                    self.active_trades[ref] = trade_record
                    self.trade_status[ref] = TradeStatus.COMPLETED
                    executed_trades[ref] = trade_record
                    
                    logger.info(f"Trade {ref} executed successfully: {trade_record}")
                    return trade_record
                    
                elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                    # Price changed, retry immediately
                    logger.warning(f"Requote for trade {ref}, retrying immediately")
                    continue
                    
                elif result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                    # FOK failed, try IOC
                    logger.warning(f"FOK failed for trade {ref}, trying IOC")
                    request["type_filling"] = mt5.ORDER_FILLING_IOC
                    result = mt5.order_send(request)
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        # Success with IOC
                        trade_record = {
                            'ref': ref,
                            'ticket': result.order,
                            'symbol': TRADING_SYMBOL,
                            'original_symbol': trade_info['Symbol'],
                            'type': trade_type,
                            'volume': result.volume,  # Important: Use actual filled volume
                            'open_price': result.price,
                            'magic': int(ref),
                            'execution_time': datetime.now().timestamp()
                        }
                        
                        self.active_trades[ref] = trade_record
                        self.trade_status[ref] = TradeStatus.COMPLETED
                        executed_trades[ref] = trade_record
                        
                        logger.info(f"Trade {ref} executed with IOC: {trade_record}")
                        return trade_record
                
                # Other errors, increment retry counter
                self.retry_counts[ref] += 1
                self.trade_status[ref] = TradeStatus.RETRYING
                logger.warning(f"Trade {ref} failed (attempt {self.retry_counts[ref]}): {result.comment}")
                time.sleep(RETRY_DELAY)
                
            except Exception as e:
                logger.error(f"Error executing trade {ref}: {e}")
                self.retry_counts[ref] += 1
                self.trade_status[ref] = TradeStatus.RETRYING
                time.sleep(RETRY_DELAY)
        
        # All retries failed
        logger.error(f"Failed to execute trade {ref} after {MAX_RETRIES} attempts")
        self.trade_status[ref] = TradeStatus.FAILED
        failed_trades.add(ref)
        return None
    
    def check_auto_trading_enabled(self):
        """Check if auto trading is enabled in MT5"""
        try:
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                logger.error("Failed to get terminal info")
                return False
            
            if not terminal_info.trade_allowed:
                logger.error("AutoTrading is DISABLED in MT5 terminal!")
                logger.error("Please enable AutoTrading in MT5:")
                logger.error("1. Go to Tools -> Options -> Expert Advisors")
                logger.error("2. Check 'Allow automated trading'")
                logger.error("3. Or click the 'AutoTrading' button in MT5 toolbar (should be green)")
                return False
            
            logger.info("AutoTrading is enabled in MT5 terminal")
            return True
            
        except Exception as e:
            logger.error(f"Error checking auto trading status: {e}")
            return False
    
    def check_for_manually_closed_trades(self):
        """Check if any active trades were manually closed in MT5"""
        if not self.active_trades:
            return
        
        try:
            # Get all current positions
            positions = mt5.positions_get()
            current_magics = set()
            
            if positions:
                current_magics = {pos.magic for pos in positions}
            
            # Check which active trades no longer have positions
            manually_closed = []
            for ref, trade_info in list(self.active_trades.items()):
                magic = int(ref)
                if magic not in current_magics:
                    manually_closed.append(ref)
            
            # Handle manually closed trades
            for ref in manually_closed:
                logger.warning(f"Trade REF {ref} was manually closed in MT5 - removing from active trades")
                self.manually_closed_refs.add(ref)
                del self.active_trades[ref]
                
        except Exception as e:
            logger.error(f"Error checking for manually closed trades: {e}")
    
    def list_all_existing_positions(self):
        """List all existing positions in MT5 for debugging"""
        try:
            logger.info("=== LISTING ALL EXISTING MT5 POSITIONS ===")
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                logger.info("No existing positions found in MT5")
                return
            
            logger.info(f"Found {len(positions)} existing positions:")
            for i, pos in enumerate(positions, 1):
                logger.info(f"  Position {i}: Symbol={pos.symbol}, Magic={pos.magic}, Comment='{pos.comment}', Volume={pos.volume}, Type={'BUY' if pos.type == 0 else 'SELL'}")
            
            logger.info("=== END OF EXISTING POSITIONS ===")
            
        except Exception as e:
            logger.error(f"Error listing existing positions: {e}")
    
    def is_duplicate_trade(self, ref):
        """Check if this REF already has an OPEN position in MT5 by checking trade comments"""
        try:
            # Get all current OPEN positions
            positions = mt5.positions_get()
            if positions is None:
                logger.info(f"✓ REF {ref} - No open positions found, can execute")
                return False  # No positions, so no duplicates
            
            logger.info(f"🔍 DUPLICATE CHECK for REF {ref} - Checking {len(positions)} open positions")
            
            # Check if any OPEN position has this REF in its comment or magic number
            for i, pos in enumerate(positions, 1):
                logger.info(f"  Position {i}: Magic={pos.magic}, Comment='{pos.comment}', Symbol={pos.symbol}")
                
                # Check magic number first (primary method)
                if pos.magic == int(ref):
                    logger.warning(f"❌ DUPLICATE DETECTED - REF {ref} already exists with Magic Number {pos.magic}")
                    logger.warning(f"   Existing position: {pos.symbol}, Volume: {pos.volume}, Magic: {pos.magic}")
                    return True
                
                # Also check comment as backup
                if pos.comment and ref in pos.comment:
                    logger.warning(f"❌ DUPLICATE DETECTED - REF {ref} found in comment: '{pos.comment}'")
                    logger.warning(f"   Existing position: {pos.symbol}, Volume: {pos.volume}")
                    return True
            
            # No open position found with this REF
            logger.info(f"✅ REF {ref} - No duplicate found, can execute trade")
            return False
            
        except Exception as e:
            logger.error(f"Error checking for duplicate trades: {e}")
            logger.info(f"⚠️ REF {ref} - Error during duplicate check, allowing trade to proceed")
            return False  # If we can't check, allow the trade
        
    def initialize_mt5(self):
        """Initialize MT5 connection with retry logic"""
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not mt5.initialize():
                    logger.error(f"MT5 initialization failed! Error code: {mt5.last_error()}")
                    retry_count += 1
                    time.sleep(2)  # Minimal delay for retry
                    continue
                
                logger.info("MT5 initialized successfully")
                account_info = mt5.account_info()
                if account_info:
                    logger.info(f"Account: {account_info.login}")
                    logger.info(f"Balance: {account_info.balance}")
                    logger.info(f"Equity: {account_info.equity}")
                
                # Check if auto trading is enabled
                if not self.check_auto_trading_enabled():
                    logger.error("Cannot proceed without AutoTrading enabled")
                    retry_count += 1
                    time.sleep(2)  # Minimal delay for retry
                    continue
                
                # Ensure Bitcoin symbol is available
                symbol_info = mt5.symbol_info(TRADING_SYMBOL)
                if symbol_info is None:
                    logger.error(f"Bitcoin symbol {TRADING_SYMBOL} not found!")
                    retry_count += 1
                    time.sleep(2)  # Minimal delay for retry
                    continue
                
                if not symbol_info.visible:
                    logger.info(f"Making Bitcoin symbol {TRADING_SYMBOL} visible...")
                    if not mt5.symbol_select(TRADING_SYMBOL, True):
                        logger.error(f"Failed to select Bitcoin symbol {TRADING_SYMBOL}")
                        retry_count += 1
                        time.sleep(2)  # Minimal delay for retry
                        continue
                
                logger.info(f"BITCOIN symbol {TRADING_SYMBOL} is ready for trading")
                self.mt5_connected = True
                return True
                
            except Exception as e:
                logger.error(f"Error initializing MT5: {e}")
                retry_count += 1
                time.sleep(2)  # Minimal delay for retry
        
        logger.error("Failed to initialize MT5 after maximum retries")
        return False
    
    def ensure_mt5_connection(self):
        """Ensure MT5 is connected, reconnect if necessary"""
        try:
            # Test connection
            account_info = mt5.account_info()
            if account_info is None:
                logger.warning("MT5 connection lost, attempting to reconnect...")
                self.mt5_connected = False
                return self.initialize_mt5()
            return True
        except Exception as e:
            logger.error(f"Error checking MT5 connection: {e}")
            self.mt5_connected = False
            return self.initialize_mt5()

    def close_trade_with_retry(self, ref: str) -> bool:
        """Close a trade with advanced retry logic and error handling"""
        self.trade_status[ref] = TradeStatus.PENDING
        self.retry_counts[ref] = 0
        
        while self.retry_counts[ref] < MAX_RETRIES:
            try:
                # Ensure connection before each attempt
                if not connection_monitor.ensure_connection():
                    logger.error(f"No connection available to close trade {ref}")
                    time.sleep(RETRY_DELAY)
                    self.retry_counts[ref] += 1
                    continue
                
                # Check if trade exists
                if ref not in self.active_trades:
                    logger.warning(f"No active trade found for REF: {ref}")
                    return False
                
                trade = self.active_trades[ref]
                
                # Get all open positions
                positions = mt5.positions_get()
                if positions is None:
                    logger.warning("No positions found")
                    return False
                
                # Find position by magic number (REF)
                target_position = None
                for pos in positions:
                    if pos.magic == int(ref):
                        target_position = pos
                        logger.info(f"Found position to close: Magic={pos.magic}, Volume={pos.volume}")
                        break
                
                if target_position is None:
                    logger.warning(f"Position with REF {ref} not found - may have been manually closed")
                    self.manually_closed_refs.add(ref)
                    del self.active_trades[ref]
                    return False
                
                self.trade_status[ref] = TradeStatus.EXECUTING
                
                # Get current price with slippage protection
                tick = mt5.symbol_info_tick(target_position.symbol)
                if not tick:
                    raise Exception("Failed to get current price")
                
                price = tick.bid if target_position.type == mt5.ORDER_TYPE_BUY else tick.ask
                
                # Prepare close request with FOK filling
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": target_position.symbol,
                    "volume": target_position.volume,
                    "type": mt5.ORDER_TYPE_SELL if target_position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                    "position": target_position.ticket,
                    "price": price,
                    "deviation": 10,  # Tight deviation for accurate closing
                    "magic": int(ref),
                    "comment": f"CLOSE-REF{ref}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_FOK,
                }
                
                # Execute close
                result = mt5.order_send(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Success!
                    logger.info(f"Successfully closed trade {ref}")
                    del self.active_trades[ref]
                    self.trade_status[ref] = TradeStatus.COMPLETED
                    return True
                    
                elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                    # Price changed, retry immediately
                    logger.warning(f"Requote while closing trade {ref}, retrying immediately")
                    continue
                    
                elif result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                    # FOK failed, try IOC
                    logger.warning(f"FOK failed while closing trade {ref}, trying IOC")
                    request["type_filling"] = mt5.ORDER_FILLING_IOC
                    result = mt5.order_send(request)
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        # Success with IOC
                        logger.info(f"Successfully closed trade {ref} with IOC")
                        del self.active_trades[ref]
                        self.trade_status[ref] = TradeStatus.COMPLETED
                        return True
                
                # Other errors, increment retry counter
                self.retry_counts[ref] += 1
                self.trade_status[ref] = TradeStatus.RETRYING
                logger.warning(f"Close attempt {self.retry_counts[ref]} failed for trade {ref}: {result.comment}")
                time.sleep(RETRY_DELAY)
                
            except Exception as e:
                logger.error(f"Error closing trade {ref}: {e}")
                self.retry_counts[ref] += 1
                self.trade_status[ref] = TradeStatus.RETRYING
                time.sleep(RETRY_DELAY)
        
        # All retries failed
        logger.error(f"Failed to close trade {ref} after {MAX_RETRIES} attempts")
        self.trade_status[ref] = TradeStatus.FAILED
        failed_trades.add(ref)
        return False

    def sync_existing_mt5_positions(self):
        """Sync existing MT5 positions into active trades tracking on startup"""
        try:
            logger.info("=== SYNCING EXISTING MT5 POSITIONS ===")
            positions = mt5.positions_get()
            
            if positions is None or len(positions) == 0:
                logger.info("No existing positions found in MT5")
                return
            
            synced_count = 0
            for pos in positions:
                try:
                    # Try to get REF from magic number first
                    ref = str(pos.magic)
                    
                    # If magic is 0, try to extract from comment
                    if pos.magic == 0 and pos.comment:
                        # Try to find REF in comment (assuming format like "REF123" or similar)
                        import re
                        ref_match = re.search(r'REF(\d+)', pos.comment)
                        if ref_match:
                            ref = ref_match.group(1)
                        else:
                            logger.warning(f"Could not extract REF from position: Magic={pos.magic}, Comment={pos.comment}")
                            continue
                    
                    # Create trade record
                    trade_record = {
                        'ref': ref,
                        'ticket': pos.ticket,
                        'symbol': pos.symbol,
                        'original_symbol': pos.symbol,
                        'type': 'Buy' if pos.type == mt5.ORDER_TYPE_BUY else 'Sell',
                        'volume': pos.volume,
                        'open_price': pos.price_open,
                        'magic': int(ref),
                        'execution_time': datetime.fromtimestamp(pos.time).timestamp()
                    }
                    
                    # Add to active trades
                    self.active_trades[ref] = trade_record
                    executed_trades[ref] = trade_record
                    synced_count += 1
                    logger.info(f"Synced position: REF={ref}, Type={trade_record['type']}, Volume={pos.volume}")
                    
                except Exception as e:
                    logger.error(f"Error syncing position {pos.ticket}: {e}")
                    continue
            
            logger.info(f"Successfully synced {synced_count} existing positions")
            logger.info("=== SYNC COMPLETE ===")
            
        except Exception as e:
            logger.error(f"Error syncing MT5 positions: {e}")

def get_current_trade_data():
    """Get current trade data from received POST data"""
    global received_trade_data
    
    with data_lock:
        return received_trade_data.copy()  # Return a copy to avoid thread issues

class ProcessMemoryMonitor:
    def __init__(self):
        self.monitoring = True
        self.monitored_processes = MONITORED_PROCESSES
        self.max_memory_percent = MAX_MEMORY_PERCENT
        self.warning_threshold = MEMORY_WARNING_THRESHOLD
        
    def get_process_memory_info(self, process_name):
        """Get memory usage for a specific process"""
        try:
            for proc in psutil.process_iter(['name', 'memory_percent']):
                if proc.info['name'] == process_name:
                    return {
                        'pid': proc.pid,
                        'memory_percent': proc.info['memory_percent'],
                        'memory_mb': proc.memory_info().rss / (1024 * 1024)  # Convert to MB
                    }
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        return None
        
    def terminate_process(self, pid):
        """Safely terminate a process"""
        try:
            process = psutil.Process(pid)
            process_name = process.name()
            logger.warning(f"Terminating process {process_name} (PID: {pid}) due to high memory usage")
            process.terminate()
            process.wait(timeout=3)  # Wait for process to terminate
            logger.info(f"Successfully terminated process {process_name}")
            return True
        except Exception as e:
            logger.error(f"Error terminating process (PID: {pid}): {e}")
            return False
            
    def monitor_processes(self):
        """Main monitoring loop"""
        logger.info("Starting process memory monitoring...")
        logger.info(f"Monitoring processes: {', '.join(self.monitored_processes)}")
        logger.info(f"Memory threshold: {self.max_memory_percent}%")
        logger.info(f"Warning threshold: {self.warning_threshold}%")
        
        while self.monitoring:
            try:
                # Get system memory info
                system_memory = psutil.virtual_memory()
                system_memory_used = system_memory.percent
                
                if system_memory_used > self.warning_threshold:
                    logger.warning(f"System memory usage high: {system_memory_used:.1f}%")
                
                # Check each monitored process
                for process_name in self.monitored_processes:
                    process_info = self.get_process_memory_info(process_name)
                    
                    if process_info:
                        memory_percent = process_info['memory_percent']
                        memory_mb = process_info['memory_mb']
                        pid = process_info['pid']
                        
                        logger.info(f"Process {process_name} (PID: {pid}) memory usage: {memory_mb:.1f}MB ({memory_percent:.1f}%)")
                        
                        if memory_percent > self.max_memory_percent:
                            logger.warning(f"Process {process_name} exceeded memory threshold ({memory_percent:.1f}% > {self.max_memory_percent}%)")
                            if self.terminate_process(pid):
                                logger.info(f"Successfully terminated {process_name} due to high memory usage")
                            else:
                                logger.error(f"Failed to terminate {process_name}")
                        elif memory_percent > self.warning_threshold:
                            logger.warning(f"Process {process_name} memory usage warning: {memory_percent:.1f}%")
                    
            except Exception as e:
                logger.error(f"Error in process memory monitoring: {e}")
            
            time.sleep(MEMORY_CHECK_INTERVAL)

def start_memory_monitor():
    """Start the memory monitoring thread"""
    monitor = ProcessMemoryMonitor()
    memory_thread = threading.Thread(target=monitor.monitor_processes, daemon=True)
    memory_thread.start()
    return monitor

def continuous_monitor_and_execute(trade_manager, check_interval=1):
    """Continuously monitor received trade data and handle new/disappeared trades - IMPROVED RELIABILITY"""
    logger.info(f"Starting reliable monitoring of received POST data (100ms intervals)")
    logger.info("SYSTEM WILL RUN INDEFINITELY - Monitor logs for activity")
    logger.info(f"ALL TRADES WILL BE EXECUTED ON BITCOIN ({TRADING_SYMBOL})")
    logger.info("RELIABLE EXECUTION MODE - Optimized for stability")
    logger.info("SYSTEM WILL CONTINUE MONITORING EVEN WHEN MARKET IS CLOSED")
    logger.info("DUPLICATE TRADE PREVENTION AND MANUAL CLOSE DETECTION ENABLED")
    logger.info("WAITING FOR POST DATA ON http://localhost:{PORT}/trades")
    
    known_refs = set()  # Track REFs we've already processed
    market_closed_warning_shown = False
    last_status_time = 0
    last_manual_check = 0
    initial_sync_done = False  # Flag to track initial sync
    
    # Start memory monitoring
    memory_monitor = start_memory_monitor()
    
    # Start UiRobot monitoring (now without command line args)
    uirobot_monitor = UiRobotMonitor()
    uirobot_thread = threading.Thread(target=uirobot_monitor.monitor_uirobot, daemon=True)
    uirobot_thread.start()
    
    logger.info("UiRobot monitoring started - will capture command line after 2 minutes")
    
    while True:  # Infinite loop - never stops
        try:
            # Update last check time
            trade_manager.last_data_check = datetime.now()
            
            # Check for manually closed trades every 5 seconds
            current_time = int(time.time())
            if current_time - last_manual_check >= 5:
                trade_manager.check_for_manually_closed_trades()
                last_manual_check = current_time
            
            # Get current REFs from received data
            current_trades = get_current_trade_data()
            current_refs = {trade['REF'] for trade in current_trades}
            
            # Special handling for first data receipt after startup
            if not initial_sync_done and current_trades:
                logger.info("First data received after startup - syncing with existing positions")
                known_refs = set(trade_manager.active_trades.keys())
                initial_sync_done = True
                logger.info(f"Initial sync complete - tracking {len(known_refs)} existing positions")
            
            # Find new trades (REFs that weren't there before)
            new_refs = current_refs - known_refs
            if new_refs:
                logger.info(f"NEW TRADES RECEIVED: {len(new_refs)} new trades: {list(new_refs)}")
                
                # Check if market is open before attempting trades
                tick = mt5.symbol_info_tick(TRADING_SYMBOL)
                if tick is None:
                    if not market_closed_warning_shown:
                        logger.warning(f"BITCOIN market appears to be closed - New trades will be queued until market opens")
                        market_closed_warning_shown = True
                else:
                    market_closed_warning_shown = False  # Reset warning when market is accessible
                
                # Execute new trades on Bitcoin INSTANTLY
                for trade in current_trades:
                    if trade['REF'] in new_refs:
                        # Double check this isn't an existing position
                        if trade['REF'] not in trade_manager.active_trades:
                            logger.info(f"Processing {trade['REF']} on BITCOIN (original symbol: {trade['Symbol']})")
                            result = trade_manager.execute_trade_with_retry(trade)
                            if result is None and tick is None:
                                logger.info(f"Trade {trade['REF']} queued - will execute when market opens")
                        else:
                            logger.info(f"Trade {trade['REF']} already exists in MT5, skipping execution")
            
            # Check which active trades are no longer in the received data
            active_refs = set(trade_manager.active_trades.keys())
            disappeared_refs = active_refs - current_refs
            
            # Only process disappeared trades after initial sync
            if initial_sync_done:
                # Close trades for disappeared REFs INSTANTLY
                for ref in disappeared_refs:
                    logger.info(f"REF {ref} no longer in received data - closing Bitcoin trade...")
                    success = trade_manager.close_trade_with_retry(ref)
                    if not success:
                        logger.warning(f"Could not close trade {ref} - may have been closed manually or market is closed")
            
            # Update known REFs
            known_refs = current_refs.copy()
            
            # Log comprehensive status every 60 seconds
            if current_time - last_status_time >= 60:
                # Check market status
                tick = mt5.symbol_info_tick(TRADING_SYMBOL)
                market_status = "OPEN" if tick is not None else "CLOSED"
                
                logger.info(f"=== SYSTEM STATUS ===")
                logger.info(f"Active Bitcoin trades: {len(trade_manager.active_trades)}")
                logger.info(f"REFs in received data: {len(current_refs)}")
                logger.info(f"Manually closed REFs: {len(trade_manager.manually_closed_refs)}")
                logger.info(f"Market status: {market_status}")
                logger.info(f"Initial sync status: {'COMPLETE' if initial_sync_done else 'PENDING'}")
                logger.info(f"Flask server: http://localhost:{PORT}")
                logger.info(f"UiRobot status: {'RUNNING' if uirobot_monitor.is_uirobot_running() else 'NOT RUNNING'}")
                logger.info(f"===================")
                last_status_time = current_time
            
            # Ultra-tiny sleep to prevent 100% CPU usage while maintaining ultra-fast response
            time.sleep(0.1)  # 100ms - more reasonable for MT5 processing and system stability
            
        except KeyboardInterrupt:
            # Even if user tries to interrupt, continue running
            logger.warning("Interrupt detected but system will CONTINUE RUNNING")
            time.sleep(0.0001)  # Ultra-tiny delay to prevent CPU overload
            continue
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            logger.info("Continuing monitoring after error...")
            time.sleep(0.0001)  # Ultra-tiny delay to prevent CPU overload
            continue

def display_trades(trades):
    """Display the extracted trade information in a formatted way"""
    if not trades:
        logger.info("No trades found!")
        return
    
    print(f"\n{'='*80}")
    print(f"{'BITCOIN TRADING SYSTEM - RECEIVED TRADE ANALYSIS':^80}")
    print(f"{'='*80}")
    print(f"ALL TRADES WILL BE EXECUTED ON BITCOIN ({TRADING_SYMBOL})")
    print(f"INSTANT EXECUTION MODE - NO DELAYS")
    print(f"Total trades received: {len(trades)}")
    print(f"{'='*80}")
    
    # Header
    print(f"{'REF':<10} {'Type':<6} {'Bought Lot':<12} {'Sold Lot':<12} {'Orig Symbol':<10} {'Timestamp':<20}")
    print(f"{'-'*80}")
    
    # Trade details
    for trade in trades:
        print(f"{trade['REF']:<10} {trade['Trade_Type']:<6} {trade['Bought_Lot']:<12.2f} {trade['Sold_Lot']:<12.2f} {trade['Symbol']:<10} {trade['Timestamp']:<20}")
    
    print(f"{'-'*80}")
    print(f"Note: All trades above will be executed on BITCOIN ({TRADING_SYMBOL}) regardless of original symbol")

def cleanup_files():
    """Cleanup all lock and state files on exit"""
    try:
        files_to_cleanup = [LOCK_FILE, PORT_LOCK_FILE, CMDLINE_FILE]
        for file in files_to_cleanup:
            if os.path.exists(file):
                os.remove(file)
    except:
        pass

def main():
    """Main execution function - NEVER STOPS - ULTRA-FAST (0.1ms)"""
    # Register cleanup for all files
    atexit.register(cleanup_files)
    
    # Ensure single instance
    if not ensure_single_instance():
        sys.exit(1)
        
    # Ensure port 3000 is available
    if not ensure_port_available():
        logger.error("Could not secure port 3000")
        sys.exit(1)
        
    print("\n" + "="*70)
    print("=== MT5 CONTINUOUS BITCOIN TRADING SYSTEM (Flask POST) ===")
    print("="*70)
    
    # Parse arguments first
    args = parse_arguments()
    
    logger.info("=== MT5 CONTINUOUS BITCOIN TRADING SYSTEM (Flask POST) ===")
    logger.info("THIS SYSTEM RUNS INDEFINITELY")
    logger.info(f"ALL TRADES WILL BE EXECUTED ON BITCOIN ({TRADING_SYMBOL})")
    logger.info("ULTRA-FAST EXECUTION MODE - 0.1ms INTERVALS")
    logger.info(f"LISTENING FOR POST DATA ON http://localhost:{PORT}/trades")
    logger.info(f"LOT PERCENTAGE: {args.lot_percentage}% (configured via C# GUI)")

    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=start_flask_server, daemon=True)
    flask_thread.start()
    logger.info("Flask server started on http://localhost:{PORT}")
    
    # Wait a moment for Flask to start
    time.sleep(2)
    
    # Outer infinite loop for system restart
    while True:
        try:
            # Initialize MT5 Trade Manager with lot percentage from command line
            trade_manager = MT5TradeManager(args.lot_percentage / 100.0)
            
            print(f"\nLOT SIZE CONFIGURATION: {args.lot_percentage}% (from C# GUI)")
            print("=" * 50)
            logger.info(f"Using {args.lot_percentage}% lot execution (configured via C# GUI)")
            break  # Exit the configuration loop
            
        except Exception as e:
            logger.error(f"Error in configuration: {e}")
            print("Error occurred, using default 100%")
            trade_manager = MT5TradeManager(1.0)
            break
    
    # Now start the main system loop
    while True:  # System restart loop
        try:
            print("\n" + "="*50)
            print("INITIALIZING MT5 CONNECTION...")
            print("="*50)
            
            # Initialize MT5 - keep trying until successful
            while not trade_manager.initialize_mt5():
                logger.error("Failed to initialize MT5, retrying instantly...")
                time.sleep(0.1)  # 100ms delay for MT5 connection issues only
            
            # IMPORTANT: Sync existing MT5 positions first
            trade_manager.sync_existing_mt5_positions()
            
            # List any existing positions for debugging
            trade_manager.list_all_existing_positions()
            
            # Check if there's any initial data
            initial_trades = get_current_trade_data()
            
            if initial_trades:
                # Display found trades
                display_trades(initial_trades)
                
                # Ask user confirmation for initial execution
                try:
                    print(f"\nFound {len(initial_trades)} initial trades to execute ON BITCOIN.")
                    confirm = input("Execute initial trades on BITCOIN? (y/n, default y): ")
                    
                    if not confirm.strip() or confirm.lower() == 'y':
                        logger.info("=== EXECUTING INITIAL TRADES ON BITCOIN ===")
                        executed_count = 0
                        skipped_count = 0
                        
                        for i, trade in enumerate(initial_trades, 1):
                            # Skip if we already have this trade from MT5 sync
                            if trade['REF'] in trade_manager.active_trades:
                                logger.info(f"Trade {trade['REF']} already exists in MT5, skipping")
                                skipped_count += 1
                                continue
                                
                            logger.info(f"--- Processing trade {i}/{len(initial_trades)}: REF {trade['REF']} ---")
                            result = trade_manager.execute_trade_with_retry(trade)
                            if result:
                                executed_count += 1
                                logger.info(f"✅ Trade {i}: REF {trade['REF']} EXECUTED successfully")
                            else:
                                skipped_count += 1
                                logger.warning(f"❌ Trade {i}: REF {trade['REF']} SKIPPED or FAILED")
                            # ULTRA-FAST - NO DELAY between initial trades
                        
                        logger.info(f"=== INITIAL TRADE SUMMARY ===")
                        logger.info(f"Total trades found: {len(initial_trades)}")
                        logger.info(f"Successfully executed: {executed_count}")
                        logger.info(f"Skipped/Failed: {skipped_count}")
                        logger.info(f"============================")
                    else:
                        logger.info("Initial Bitcoin trade execution skipped")
                        
                except KeyboardInterrupt:
                    logger.info("Using default settings and continuing...")
            else:
                logger.info("No initial trades found")
                logger.info("System will monitor for POST data on http://localhost:{PORT}/trades")
            
            # Start continuous monitoring (NEVER STOPS - ULTRA-FAST)
            print("\n" + "="*50)
            print("STARTING ULTRA-FAST MONITORING (0.1ms)...")
            print("Flask server running on http://localhost:{PORT}")
            print("POST trade data to: http://localhost:{PORT}/trades")
            print("Check status at: http://localhost:{PORT}/status")
            print("="*50)
            logger.info("Starting ULTRA-FAST BITCOIN monitoring mode (0.1ms intervals)...")
            logger.info(f"Using {trade_manager.lot_percentage*100}% lot size (configured via C# GUI)")
            logger.info("Ready to receive POST data on http://localhost:{PORT}/trades")
            continuous_monitor_and_execute(trade_manager, 1)
            
        except Exception as e:
            logger.error(f"System error occurred: {e}")
            logger.info("Restarting system instantly...")
            time.sleep(0.1)  # 100ms delay for system restart only
            continue

# Add new endpoint for dynamic lot size updates
@app.route('/update_lot_size', methods=['POST'])
def update_lot_size():
    """Update lot size without restart"""
    try:
        data = request.get_json()
        if not data or 'lot_percentage' not in data:
            return jsonify({
                "status": "error",
                "message": "Missing lot_percentage in request"
            }), 400
            
        new_percentage = float(data['lot_percentage'])
        if not 0 < new_percentage <= 100:
            return jsonify({
                "status": "error",
                "message": f"Invalid lot percentage {new_percentage}. Must be between 0 and 100"
            }), 400
            
        # Get trade manager instance
        trade_manager = get_trade_manager()
        if trade_manager.update_lot_size(new_percentage):
            return jsonify({
                "status": "success",
                "message": f"Lot size updated to {new_percentage}%"
            }), 200
        else:
            return jsonify({
                "status": "error",
                "message": "Failed to update lot size"
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# Global trade manager instance
_trade_manager = None

def get_trade_manager():
    """Get or create trade manager instance"""
    global _trade_manager
    if _trade_manager is None:
        args = parse_arguments()
        _trade_manager = MT5TradeManager(args.lot_percentage / 100.0)
    return _trade_manager

def kill_process_on_port(port):
    """Kill any process using the specified port"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.connections()
                for conn in connections:
                    if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == port:
                        logger.warning(f"Found process {proc.name()} (PID: {proc.pid}) using port {port}")
                        if proc.pid != os.getpid():  # Don't kill ourselves
                            proc.kill()
                            proc.wait(timeout=5)
                            logger.info(f"Killed process {proc.name()} to free port {port}")
                            return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
    except Exception as e:
        logger.error(f"Error killing process on port {port}: {e}")
    return False

def ensure_port_available():
    """Ensure port 3000 is available, killing any process using it"""
    try:
        # Check if port is in use
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', PORT))
        sock.close()
        
        if result == 0:  # Port is in use
            logger.warning(f"Port {PORT} is in use, attempting to free it")
            if kill_process_on_port(PORT):
                time.sleep(1)  # Wait for port to be fully released
                
        # Create port lock file
        with open(PORT_LOCK_FILE, 'w') as f:
            f.write(str(os.getpid()))
            
        # Register cleanup
        atexit.register(cleanup_port_lock)
        
        return True
        
    except Exception as e:
        logger.error(f"Error ensuring port availability: {e}")
        return False

def cleanup_port_lock():
    """Remove port lock file on exit"""
    try:
        if os.path.exists(PORT_LOCK_FILE):
            os.remove(PORT_LOCK_FILE)
    except:
        pass

def check_port_owner():
    """Check if we own the port lock"""
    try:
        if os.path.exists(PORT_LOCK_FILE):
            with open(PORT_LOCK_FILE, 'r') as f:
                pid = int(f.read().strip())
                if pid == os.getpid():
                    return True
                try:
                    # Check if process exists
                    proc = psutil.Process(pid)
                    if proc.is_running():
                        logger.error(f"Port 3000 is owned by another process (PID: {pid})")
                        return False
                except psutil.NoSuchProcess:
                    # Process doesn't exist, we can take over
                    pass
        return True
    except Exception as e:
        logger.error(f"Error checking port owner: {e}")
        return False

class PortMonitor(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.monitoring = True
        
    def run(self):
        """Monitor port 3000 and ensure we maintain exclusive access"""
        while self.monitoring:
            try:
                # Check if we still own the port
                if not check_port_owner():
                    logger.error("Lost port 3000 ownership!")
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                
                # Check if any other process is using our port
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                result = sock.connect_ex(('127.0.0.1', PORT))
                sock.close()
                
                if result == 0:  # Port is in use
                    # Check if it's us using the port
                    our_port = False
                    for conn in psutil.Process(os.getpid()).connections():
                        if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == PORT:
                            our_port = True
                            break
                    
                    if not our_port:
                        logger.warning("Another process is using port 3000!")
                        kill_process_on_port(PORT)
                
            except Exception as e:
                logger.error(f"Error in port monitor: {e}")
                
            time.sleep(1)  # Check every second

if __name__ == "__main__":
    main() 
