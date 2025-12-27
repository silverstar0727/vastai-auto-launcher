import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_pitch(ax, field_dim=(105, 68), line_color='white', pitch_color='#1b4d3e', show_grid=False):
    """
    Draws a football pitch on the given axes.
    """
    length, width = field_dim
    
    # Grid
    if show_grid:
        # Draw grid lines every ~10 meters or so
        for x in range(0, int(length)+10, 10):
            ax.plot([x, x], [0, width], color='lightgray', linewidth=1, alpha=0.6, linestyle='--')
        for y in range(0, int(width)+10, 10):
            ax.plot([0, length], [y, y], color='lightgray', linewidth=1, alpha=0.6, linestyle='--')

    # Pitch Outline & Centre Line
    ax.plot([0, 0], [0, width], color=line_color, linewidth=2)
    ax.plot([0, length], [width, width], color=line_color, linewidth=2)
    ax.plot([length, length], [width, 0], color=line_color, linewidth=2)
    ax.plot([length, 0], [0, 0], color=line_color, linewidth=2)
    ax.plot([length/2, length/2], [0, width], color=line_color, linewidth=2)
    
    # Penalty Areas
    def draw_rect(x_start, y_start, w, h):
        ax.plot([x_start, x_start], [y_start, y_start+h], color=line_color, linewidth=2)
        ax.plot([x_start, x_start+w], [y_start+h, y_start+h], color=line_color, linewidth=2)
        ax.plot([x_start+w, x_start+w], [y_start+h, y_start], color=line_color, linewidth=2)
        ax.plot([x_start+w, x_start], [y_start, y_start], color=line_color, linewidth=2)

    # Left Penalty Area (approx dimensions based on standard)
    penalty_box_width = 16.5
    penalty_box_height = 40.32 # 16.5*2 + 7.32
    penalty_box_y_start = (width - penalty_box_height) / 2
    
    draw_rect(0, penalty_box_y_start, penalty_box_width, penalty_box_height)
    draw_rect(length-penalty_box_width, penalty_box_y_start, penalty_box_width, penalty_box_height)
    
    # 6 yard boxes (Goal Area)
    goal_box_width = 5.5
    goal_box_height = 18.32 # 5.5*2 + 7.32
    goal_box_y_start = (width - goal_box_height) / 2
    
    draw_rect(0, goal_box_y_start, goal_box_width, goal_box_height)
    draw_rect(length-goal_box_width, goal_box_y_start, goal_box_width, goal_box_height)

    # Centre Circle
    circle = plt.Circle((length/2, width/2), 9.15, color=line_color, fill=False, linewidth=2)
    ax.add_artist(circle)
    
    # Centre Spot
    ax.scatter(length/2, width/2, color=line_color, s=20)
    
    # Penalty Spots
    ax.scatter(11, width/2, color=line_color, s=20)
    ax.scatter(length-11, width/2, color=line_color, s=20)
    
    # Goals
    ax.plot([0, 0], [(width/2)-3.66, (width/2)+3.66], color=line_color, linewidth=4) # Left goal
    ax.plot([length, length], [(width/2)-3.66, (width/2)+3.66], color=line_color, linewidth=4) # Right goal

    ax.set_facecolor(pitch_color)
    ax.set_xlim(-5, length+5)
    ax.set_ylim(-5, width+5)
    ax.set_aspect('equal')
    ax.axis('off')

def plot_episode(episode_data, ax=None, title="Episode", pred_x=None, pred_y=None):
    """
    Plots a single episode sequence.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        
    draw_pitch(ax, show_grid=True)
    
    # Sort by action index
    episode_data = episode_data.sort_values('action_id')
    
    # Color by team
    teams = episode_data['team_id'].unique()
    colors = {team: c for team, c in zip(teams, ['red', 'blue'])}
    if len(colors) == 0:
        # Fallback if no team_id
        colors = {}
        
    for i, (_, row) in enumerate(episode_data.iterrows()):
        team = row.get('team_id')
        color = colors.get(team, 'yellow')
        
        # Check if it is the last event
        is_last_event = (i == len(episode_data) - 1)
        if is_last_event:
            color = '#00FF00' # Lime green for the result/last event
            edge_color = 'black'
            z_order = 20
        else:
            edge_color = 'white'
            z_order = 10
        
        # Start point
        ax.scatter(row['start_x'], row['start_y'], color=color, s=120 if is_last_event else 80, 
                   edgecolors=edge_color, linewidths=2 if is_last_event else 1, zorder=z_order)
        
        # Improved Labeling: Show sequence number AND Action Type
        label_text = f"{i+1}. {row['type_name']}"
        ax.text(row['start_x']+1, row['start_y']+1, label_text, color='white', 
                ha='left', va='bottom', fontsize=8 if is_last_event else 7, 
                fontweight='bold' if is_last_event else 'normal', zorder=z_order+1,
                bbox=dict(facecolor='black', alpha=0.5 if is_last_event else 0.3, edgecolor='none', pad=1))
        
        # Arrow if moved
        if pd.notna(row['end_x']) and pd.notna(row['end_y']):
            if (row['start_x'] != row['end_x']) or (row['start_y'] != row['end_y']):
                ax.arrow(
                    row['start_x'], row['start_y'], 
                    row['end_x'] - row['start_x'], row['end_y'] - row['start_y'], 
                    head_width=1.0, head_length=1.5, fc=color, ec=color, alpha=0.6, length_includes_head=True
                )

    # Plot Prediction if provided
    if pred_x is not None and pred_y is not None:
         ax.scatter(pred_x, pred_y, color='red', s=150, marker='X', zorder=30, label='Prediction', edgecolors='white', linewidths=2)
         ax.legend(loc='upper right')

    ax.set_title(title + f" (Episode: {episode_data.iloc[0]['game_episode']})", fontsize=15)
    return ax
