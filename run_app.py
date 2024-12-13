import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dashboard.app import create_app

app = create_app()
app.run_server(debug=True)