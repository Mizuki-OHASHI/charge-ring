"""
Make matplotlib use WebAgg backend when under SSH connection.
"""

import os
import atexit
import matplotlib as mpl

def is_under_ssh_connection():
    # The environment variable `SSH_CONNECTION` exists only in the SSH session.
    return 'SSH_CONNECTION' in os.environ.keys()

def use_WebAgg(port = 8000, port_retries = 50):
    """
    Use WebAgg for matplotlib backend.
    """
    current_backend = mpl.get_backend()
    current_webagg_configs = {
        'port': mpl.rcParams['webagg.port'],
        'port_retries': mpl.rcParams['webagg.port_retries'],
        'open_in_browser': mpl.rcParams['webagg.open_in_browser']
    }
    def reset():
        mpl.use(current_backend)
        mpl.rc('webagg', **current_webagg_configs)
    
    mpl.rc('webagg', **{
        'port': port,
        'port_retries': port_retries,
        'open_in_browser': False
    })
    mpl.use('WebAgg')
    atexit.register(reset)


if is_under_ssh_connection():
    use_WebAgg(port = 8000, port_retries = 50)