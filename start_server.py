#!/usr/bin/env python3
"""
Startup script for Render deployment.
This ensures the PORT environment variable is properly used.
"""
import os
import sys

def main():
    # Get PORT from environment, default to 10000 if not set
    port = os.environ.get('PORT', '10000')
    host = '0.0.0.0'
    
    # Validate port is a number
    try:
        int(port)
    except ValueError:
        print(f"ERROR: PORT '{port}' is not a valid number. Using default 10000.")
        port = '10000'
    
    print("=" * 60)
    print("Email Tone Checker - Starting Server")
    print("=" * 60)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Python: {sys.version}")
    print("=" * 60)
    sys.stdout.flush()
    
    # Start Gunicorn with proper port binding
    cmd = [
        'gunicorn',
        'app:app',
        '--bind', f'{host}:{port}',
        '--workers', '1',
        '--timeout', '120',
        '--keep-alive', '5',
        '--access-logfile', '-',
        '--error-logfile', '-',
        '--log-level', 'info'
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    sys.stdout.flush()
    
    # Execute Gunicorn (this replaces the current process)
    try:
        os.execvp('gunicorn', cmd)
    except Exception as e:
        print(f"ERROR: Failed to start Gunicorn: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

