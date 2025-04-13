import socket
import time

def measure_ping(host="api.openai.com", port=443, count=5):
    """
    Measures the approximate network latency (ping) by timing a TCP handshake.
    This method avoids using system-dependent 'ping' commands.
    """
    latencies = []

    for _ in range(count):
        try:
            # Create a socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # Set timeout for connection attempt
            
            start_time = time.time()  # Start timing
            sock.connect((host, port))  # Open TCP connection
            end_time = time.time()  # End timing
            
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)

            sock.close()  # Close connection
        except socket.error:
            latencies.append(None)  # Store None if the connection fails

    # Filter out failed attempts
    latencies = [lat for lat in latencies if lat is not None]

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        return avg_latency
    else:
        return None 