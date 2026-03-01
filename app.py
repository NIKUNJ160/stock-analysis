import streamlit as st
import subprocess
import os
import time

st.set_page_config(page_title="Quant Dashboard", layout="wide")

st.title("Live Quantitative Trading Architecture")
st.markdown("Monitor and control your trading algorithms.")

# Side panel for controls
st.sidebar.header("Controls")

# Start/Stop Engine logic
if "engine_process" not in st.session_state:
    st.session_state.engine_process = None

def start_engine():
    if st.session_state.engine_process is None:
        venv_python = os.path.join("venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join("venv", "bin", "python")
        if not os.path.exists(venv_python):
            venv_python = "python"  # Fallback
            
        st.session_state.engine_process = subprocess.Popen(
            [venv_python, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        st.sidebar.success("Engine started!")

def stop_engine():
    if st.session_state.engine_process is not None:
        st.session_state.engine_process.terminate()
        st.session_state.engine_process = None
        st.sidebar.warning("Engine stopped!")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("Start Engine", type="primary"):
        start_engine()
with col2:
    if st.button("Stop Engine"):
        stop_engine()

# Main displays
col_status, col_logs = st.columns([1, 2])

with col_status:
    st.subheader("System Status")
    if st.session_state.engine_process is None:
        st.error("Engine is OFFLINE")
    else:
        st.success("Engine is RUNNING")

with col_logs:
    st.subheader("Live Logs")
    
    # Check if we need to start the log reader thread
    if st.session_state.engine_process is not None:
        if "log_queue" not in st.session_state:
            import queue
            import threading
            st.session_state.log_queue = queue.Queue(maxsize=100)
            
            def read_logs(proc, q):
                for line in iter(proc.stdout.readline, ""):
                    if line:
                        if q.full():
                            try:
                                q.get_nowait()
                            except queue.Empty:
                                pass
                        q.put(line.strip())
            
            st.session_state.log_thread = threading.Thread(
                target=read_logs, 
                args=(st.session_state.engine_process, st.session_state.log_queue),
                daemon=True
            )
            st.session_state.log_thread.start()

        # Initialize or retrieve the display buffer
        if "log_buffer" not in st.session_state:
            st.session_state.log_buffer = []

        # Drain the queue into our buffer
        while not st.session_state.log_queue.empty():
            st.session_state.log_buffer.append(st.session_state.log_queue.get())
        
        # Keep only the last 30 lines
        max_lines = 30
        if len(st.session_state.log_buffer) > max_lines:
            st.session_state.log_buffer = st.session_state.log_buffer[-max_lines:]

        # Display logs
        st.code("\n".join(st.session_state.log_buffer))
        
        # Tell Streamlit to auto-rerun to keep fetching logs
        time.sleep(1)
        st.rerun()
    else:
        st.info("Start the engine to see logs here.")
        # Cleanup state if engine is off
        if "log_queue" in st.session_state:
            del st.session_state.log_queue
        if "log_thread" in st.session_state:
            del st.session_state.log_thread
        if "log_buffer" in st.session_state:
            del st.session_state.log_buffer
