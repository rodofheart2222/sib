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
import requests  # Add requests for API calls
from threading import Lock, Event
from contextlib import closing

# Setup advanced logging
logger = logging.getLogger(__name__)

# Add Xano API constants

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

# Add these global locks
trade_data_lock = Lock()
active_trades_lock = Lock()
known_refs_lock = Lock()
cleanup_lock = Lock()
shutdown_event = Event()

# Add these constants near other constants
HOST = '0.0.0.0'  # Listen on all available interfaces
SOCKET_TIMEOUT = 3  # Socket timeout in seconds
SOCKET_BACKLOG = 5  # Socket backlog for pending connections
SO_REUSEADDR = True  # Allow socket address reuse

class SocketManager:
    def __init__(self):
        self.socket_lock = Lock()
        self.bound_socket = None
        
    def bind_socket(self):
        """Bind to socket with proper error handling and address reuse"""
        with self.socket_lock:
            try:
                if self.bound_socket:
                    try:
                        self.bound_socket.close()
                    except:
                        pass
                
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.settimeout(SOCKET_TIMEOUT)
                
                # Try to bind to all interfaces first
                try:
                    sock.bind((HOST, PORT))
                    logger.info(f"Bound to all interfaces ({HOST}:{PORT})")
                except socket.error:
                    # Fall back to localhost if binding to all interfaces fails
                    try:
                        sock.bind(('127.0.0.1', PORT))
                        logger.info(f"Bound to localhost (127.0.0.1:{PORT})")
                    except socket.error as e:
                        logger.error(f"Could not bind to any interface: {e}")
                        return False
                
                sock.listen(SOCKET_BACKLOG)
                self.bound_socket = sock
                return True
                
            except Exception as e:
                logger.error(f"Error binding socket: {e}")
                return False
    
    def release_socket(self):
        """Safely release bound socket"""
        with self.socket_lock:
            if self.bound_socket:
                try:
                    self.bound_socket.close()
                except:
                    pass
                self.bound_socket = None

# Create global socket manager
socket_manager = SocketManager()

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

XANO_API_BASE_URL = "https://x8ki-letl-twmt.n7.xano.io/api:M56UdKHz"
XANO_TRADE_DATA_ENDPOINT = f"{XANO_API_BASE_URL}/trade_data"

# Add dictionary to store Xano trade IDs
xano_trade_ids = {}  # Maps REF to Xano trade ID

def post_trade_to_xano(action: str, lot_amount: float) -> Optional[int]:
    """Post trade data to Xano API and return the trade ID"""
    try:
        payload = {
            "action": action,
            "lot_amount": lot_amount
        }
        
        response = requests.post(XANO_TRADE_DATA_ENDPOINT, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data.get('id')
        
    except Exception as e:
        logger.error(f"Error posting trade to Xano: {e}")
        return None

def delete_trade_from_xano(trade_id: int) -> bool:
    """Delete trade data from Xano API"""
    try:
        delete_url = f"{XANO_TRADE_DATA_ENDPOINT}/{trade_id}"
        response = requests.delete(delete_url)
        response.raise_for_status()
        return True
        
    except Exception as e:
        logger.error(f"Error deleting trade from Xano: {e}")
        return False
def check_executor_memory():
    """Check if UiPath.Executor.exe memory exceeds 1GB"""
    try:
        for proc in psutil.process_iter(['name', 'memory_info']):
            if proc.info['name'] == 'UiPath.Executor.exe':
                memory_mb = proc.info['memory_info'].rss / (1024 * 1024)  # Convert to MB
                if memory_mb > 1024:  # More than 1GB
                    logger.warning(f"UiPath.Executor.exe using {memory_mb:.1f}MB RAM - forcing restart")
                    return True
        return False
    except Exception as e:
        logger.error(f"Error checking executor memory: {e}")
        return False

class UiRobotMonitor:
    def __init__(self):
        self.monitoring = True
        self.last_restart = 0
        self.restart_count = 0
        self.captured_cmdline = None
        
    def is_uirobot_running(self):
        """Check if UiRobot is running"""
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] == 'UiRobot.exe':
                    return True
            return False
        except:
            return False
        
    def capture_cmdline(self):
        """Continuously try to capture command line - ZERO DELAY"""
        try:
            # Always try to capture if UiRobot is running
            if self.is_uirobot_running():
                new_cmdline = get_uirobot_cmdline()
                
                if new_cmdline:
                    # Only update and save if we got new arguments
                    if new_cmdline != self.captured_cmdline:
                        self.captured_cmdline = new_cmdline
                        logger.info(f"Captured new command line arguments: {self.captured_cmdline}")
                        save_cmdline(self.captured_cmdline)
                
                # If we don't have any command line yet, try loading from file
                elif not self.captured_cmdline:
                    saved_cmdline = load_cmdline()
                    if saved_cmdline:
                        self.captured_cmdline = saved_cmdline
                        logger.info(f"Loaded saved command line: {self.captured_cmdline}")
                    
        except Exception as e:
            logger.error(f"Error capturing command line: {e}")
                
    def restart_uirobot(self):
        """Restart UiRobot.exe with captured command line - ZERO DELAY"""
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
                    logger.error("UiRobot.exe not found - retrying immediately")
                    continue  # Instant retry
                    
                # Kill all UiPath processes including UiPath.Executor.exe
                logger.info("Terminating all UiPath processes...")
                killed_processes = kill_uipath_processes()
                if killed_processes:
                    logger.info(f"Successfully terminated: {', '.join(killed_processes)}")
                
                # Use captured command line if available, otherwise try to load saved
                cmdline = self.captured_cmdline
                if not cmdline:
                    cmdline = load_cmdline()
                if not cmdline:
                    cmdline = []  # Use empty args if none found
                
                # Start new UiRobot process
                subprocess.Popen([uirobot_path] + cmdline)
                logger.info(f"Started UiRobot.exe with args: {cmdline}")
                
                # Instantly try to verify and capture
                if self.is_uirobot_running():
                    self.capture_cmdline()
                    return True
                
                logger.error("UiRobot.exe failed to start - retrying immediately")
                continue  # Instant retry
                
            except Exception as e:
                logger.error(f"Error restarting UiRobot: {e}")
                continue  # Instant retry
            
    def monitor_uirobot(self):
        """Monitor UiRobot and restart if needed - ZERO DELAY"""
        logger.info("Starting ZERO-DELAY UiRobot monitoring with memory checks...")
        
        while True:  # Monitor forever
            try:
                # Check UiPath.Executor.exe memory
                if check_executor_memory():
                    logger.warning("UiPath.Executor.exe memory exceeded 1GB - forcing restart")
                    self.restart_uirobot()
                    continue
                
                # Always try to capture command line
                self.capture_cmdline()
                
                if not self.is_uirobot_running():
                    current_time = time.time()
                    
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
                continue  # Instant retry

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
    """Start Flask server with proper socket handling"""
    logger.info(f"Starting Flask server on {HOST}:{PORT}...")
    try:
        # Ensure socket is bound
        if not socket_manager.bind_socket():
            logger.error("Could not bind socket")
            sys.exit(1)
        
        # Start port monitor
        port_monitor = PortMonitor()
        port_monitor.start()
        
        # Configure Flask to use our socket settings
        app.config['HOST'] = HOST
        app.config['PORT'] = PORT
        app.config['THREADED'] = True  # Enable threaded mode
        
        # Run Flask with our socket configuration
        app.run(
            host=HOST,
            port=PORT,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Error starting Flask server: {e}")
        logger.exception("Full traceback:")

class MT5TradeManager:
    def __init__(self, lot_percentage):
        self.active_trades = {}
        self.active_trades_lock = Lock()  # Add lock for active trades
        self.manually_closed_refs = set()
        self.manually_closed_lock = Lock()  # Add lock for manually closed refs
        self._lot_percentage = lot_percentage
        self.mt5_connected = False
        self.last_data_check = datetime.now()
        self.trade_status = {}
        self.trade_status_lock = Lock()  # Add lock for trade status
        self.retry_counts = {}
        self.last_market_check = 0
        self.market_is_open = False
        self._lot_lock = threading.Lock()
        
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
                    # Post to Xano API
                    xano_trade_id = post_trade_to_xano(
                        action=trade_type.upper(),
                        lot_amount=lot_size
                    )
                    
                    if xano_trade_id:
                        # Store Xano trade ID for later deletion
                        xano_trade_ids[ref] = xano_trade_id
                        logger.info(f"Trade {ref} posted to Xano with ID: {xano_trade_id}")
                    
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
                        'execution_time': datetime.now().timestamp(),
                        'xano_trade_id': xano_trade_id  # Store Xano ID in trade record
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
                        # Post to Xano API
                        xano_trade_id = post_trade_to_xano(
                            action=trade_type.upper(),
                            lot_amount=lot_size
                        )
                        
                        if xano_trade_id:
                            # Store Xano trade ID for later deletion
                            xano_trade_ids[ref] = xano_trade_id
                            logger.info(f"Trade {ref} posted to Xano with ID: {xano_trade_id}")
                        
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
                            'execution_time': datetime.now().timestamp(),
                            'xano_trade_id': xano_trade_id  # Store Xano ID in trade record
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
        """Thread-safe check for manually closed trades"""
        if not self.active_trades:
            return
            
        try:
            positions = mt5.positions_get()
            current_magics = set()
            
            if positions:
                current_magics = {pos.magic for pos in positions}
            
            manually_closed = []
            with self.active_trades_lock:  # Use lock when accessing active_trades
                for ref, trade_info in list(self.active_trades.items()):
                    magic = int(ref)
                    if magic not in current_magics:
                        manually_closed.append(ref)
            
            # Handle manually closed trades
            for ref in manually_closed:
                with self.manually_closed_lock:  # Use lock when modifying manually_closed_refs
                    self.manually_closed_refs.add(ref)
                with self.active_trades_lock:  # Use lock when modifying active_trades
                    if ref in self.active_trades:
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
                    
                    # Delete from Xano if we have the ID
                    if ref in xano_trade_ids:
                        xano_trade_id = xano_trade_ids[ref]
                        if delete_trade_from_xano(xano_trade_id):
                            logger.info(f"Deleted trade {ref} from Xano (ID: {xano_trade_id})")
                            del xano_trade_ids[ref]
                    
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
                    # Delete from Xano if we have the ID
                    if ref in xano_trade_ids:
                        xano_trade_id = xano_trade_ids[ref]
                        if delete_trade_from_xano(xano_trade_id):
                            logger.info(f"Deleted trade {ref} from Xano (ID: {xano_trade_id})")
                            del xano_trade_ids[ref]
                    
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
                        # Delete from Xano if we have the ID
                        if ref in xano_trade_ids:
                            xano_trade_id = xano_trade_ids[ref]
                            if delete_trade_from_xano(xano_trade_id):
                                logger.info(f"Deleted trade {ref} from Xano (ID: {xano_trade_id})")
                                del xano_trade_ids[ref]
                        
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
    """Thread-safe continuous monitoring"""
    known_refs = set()
    known_refs_lock = Lock()  # Local lock for known_refs
    
    while not shutdown_event.is_set():  # Check for shutdown
        try:
            # Get current REFs from received data
            with trade_data_lock:  # Use lock when accessing received_trade_data
                current_trades = get_current_trade_data()
            current_refs = {trade['REF'] for trade in current_trades}
            
            # Thread-safe comparison of refs
            with known_refs_lock:
                new_refs = current_refs - known_refs
            
            if new_refs:
                # Process new trades
                for trade in current_trades:
                    if trade['REF'] in new_refs:
                        with trade_manager.active_trades_lock:
                            if trade['REF'] not in trade_manager.active_trades:
                                result = trade_manager.execute_trade_with_retry(trade)
            
            # Update known refs safely
            with known_refs_lock:
                known_refs = current_refs.copy()
            
            time.sleep(0.1)  # Prevent CPU overload
            
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
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
    """Thread-safe cleanup"""
    with cleanup_lock:  # Use lock during cleanup
        try:
            files_to_cleanup = [LOCK_FILE, PORT_LOCK_FILE, CMDLINE_FILE]
            for file in files_to_cleanup:
                if os.path.exists(file):
                    os.remove(file)
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def main():
    """Main function with proper thread shutdown"""
    try:
        # Register cleanup
        atexit.register(cleanup_files)
        
        # Start threads
        threads = []
        
        # Flask thread
        flask_thread = threading.Thread(target=start_flask_server, daemon=True)
        flask_thread.start()
        threads.append(flask_thread)
        
        # Memory monitor thread
        memory_monitor = start_memory_monitor()
        
        # UiRobot monitor thread
        uirobot_monitor = UiRobotMonitor()
        uirobot_thread = threading.Thread(target=uirobot_monitor.monitor_uirobot, daemon=True)
        uirobot_thread.start()
        threads.append(uirobot_thread)
        
        # Port monitor thread
        port_monitor = PortMonitor()
        port_monitor.start()
        threads.append(port_monitor)
        
        try:
            # Main loop
            while True:
                if shutdown_event.is_set():
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Shutting down gracefully...")
            shutdown_event.set()  # Signal threads to stop
            
            # Wait for threads to finish
            for thread in threads:
                if thread.is_alive():
                    thread.join(timeout=5)
            
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        shutdown_event.set()
    finally:
        cleanup_files()

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
    """Ensure port is available with proper socket handling"""
    try:
        # First check if port is in use
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(SOCKET_TIMEOUT)
            result = sock.connect_ex(('127.0.0.1', PORT))
            
            if result == 0:  # Port is in use
                logger.warning(f"Port {PORT} is in use, attempting to free it")
                if kill_process_on_port(PORT):
                    time.sleep(1)  # Wait for port to be fully released
                    
                    # Double check port is now free
                    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as check_sock:
                        check_sock.settimeout(SOCKET_TIMEOUT)
                        result = check_sock.connect_ex(('127.0.0.1', PORT))
                        if result == 0:
                            logger.error(f"Port {PORT} still in use after attempting to free it")
                            return False
        
        # Try to bind the socket
        if not socket_manager.bind_socket():
            return False
        
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
        self.socket_check_interval = 1  # Check every second
        
    def run(self):
        """Monitor port with improved socket handling"""
        while self.monitoring:
            try:
                # Check if we still own the port
                if not check_port_owner():
                    logger.error("Lost port ownership!")
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
                
                # Check socket binding
                with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                    sock.settimeout(SOCKET_TIMEOUT)
                    result = sock.connect_ex(('127.0.0.1', PORT))
                    
                    if result == 0:  # Port is in use
                        # Check if it's us using the port
                        our_port = False
                        for conn in psutil.Process(os.getpid()).connections():
                            if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == PORT:
                                our_port = True
                                break
                        
                        if not our_port:
                            logger.warning(f"Another process is using port {PORT}!")
                            if kill_process_on_port(PORT):
                                # Try to rebind our socket
                                if not socket_manager.bind_socket():
                                    logger.error("Failed to rebind socket")
                                    os.kill(os.getpid(), signal.SIGTERM)
                                    return
                
            except Exception as e:
                logger.error(f"Error in port monitor: {e}")
                
            time.sleep(self.socket_check_interval)
    
    def stop(self):
        """Stop monitoring and release socket"""
        self.monitoring = False
        socket_manager.release_socket()

def cleanup_on_exit():
    """Cleanup function for proper shutdown"""
    try:
        # Stop port monitor
        if hasattr(cleanup_on_exit, 'port_monitor'):
            cleanup_on_exit.port_monitor.stop()
        
        # Release socket
        socket_manager.release_socket()
        
        # Remove lock files
        cleanup_files()
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

# Register cleanup function
atexit.register(cleanup_on_exit)

if __name__ == "__main__":
    main() 
