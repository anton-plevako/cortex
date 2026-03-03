import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
sys.path.insert(0, "src")

from cortex.graph import app_graph

app_graph.get_graph().draw_mermaid_png(output_file_path="cortex_flow.png")
print("Saved → cortex_flow.png")
