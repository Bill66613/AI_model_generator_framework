# Create a flowchart for the motion detection process using Seeed XIAO nRF52840 Sense
import graphviz

# Update the flowchart for a general motion detection process
motion_tracking_general = graphviz.Digraph(format='png', graph_attr={'rankdir': 'LR'})

# Add generalized nodes to represent the flow
motion_tracking_general.node("Sensor", "Motion Sensor\n(e.g., IMU, Camera)", shape="box", style="filled", color="lightblue")
motion_tracking_general.node("Preprocess", "Preprocessing\n(Noise Reduction, Normalization)", shape="ellipse", style="filled", color="lightgray")
motion_tracking_general.node("Model", "Trained AI Model\n(Motion Detection)", shape="box", style="filled", color="lightgreen")
motion_tracking_general.node("Transmission", "Data Transmission\n(BLE, Wi-Fi, etc.)", shape="ellipse", style="filled", color="lightyellow")
motion_tracking_general.node("Device", "User Device\n(Displays Motion Data)", shape="box", style="filled", color="lightpink")

# Add edges to represent the flow
motion_tracking_general.edges([
    ("Sensor", "Preprocess"),
    ("Preprocess", "Model"),
    ("Model", "Transmission"),
    ("Transmission", "Device")
])

# Render and display the updated diagram
diagram_general_path = "/mnt/data/Motion_Detection_Process_General"
motion_tracking_general.render(diagram_general_path, view=False)
diagram_general_path + ".png"

