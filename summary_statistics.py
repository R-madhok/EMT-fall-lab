import pandas as pd

df = pd.read_csv("ball_tracking.csv")
df = df[df["touching_dome"] == 0]

columns = ["gap_mm", "center_distance_m", "vertical_acceleration_m_per_s2"]
stats = df[columns].describe().T.round(5)
stats.to_csv("summary_statistics.csv", index_label="variable")
