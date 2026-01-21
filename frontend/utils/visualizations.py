"""
Visualization utilities for Spotify Wrapped
All plots use dark theme matching Spotify aesthetic
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class Visualizer:
    """Handles all data visualizations with consistent Spotify theme"""
    
    def __init__(self):
        # Spotify color palette
        self.colors = {
            'primary': '#1DB954',
            'secondary': '#1ed760',
            'background': '#191414',
            'card': '#282828',
            'text': '#FFFFFF',
            'text_secondary': '#b3b3b3'
        }
        
        # Base layout template
        self.base_layout = dict(
            plot_bgcolor=self.colors['background'],
            paper_bgcolor=self.colors['background'],
            font=dict(color=self.colors['text'], family='Helvetica Neue, sans-serif'),
            title_font=dict(size=24, color=self.colors['primary']),
            margin=dict(l=40, r=40, t=80, b=40)
        )
    
    def plot_top_artists(self, artists):
        """Horizontal bar chart of top artists"""
        # Extract data
        names = [a['artist'] for a in artists]
        counts = [a['track_count'] for a in artists]
        
        # Reverse for better display (highest at top)
        names.reverse()
        counts.reverse()
        
        fig = go.Figure(data=[
            go.Bar(
                y=names,
                x=counts,
                orientation='h',
                marker=dict(
                    color=self.colors['primary'],
                    line=dict(color=self.colors['secondary'], width=1)
                ),
                text=counts,
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Tracks: %{x}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            **self.base_layout,
            title='Your Top Artists',
            xaxis_title='Number of Tracks',
            yaxis_title='',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_temporal_trends(self, yearly, monthly):
        """Line chart showing songs added over time"""
        # Yearly trend
        years = [y['year'] for y in yearly]
        counts = [y['track_count'] for y in yearly]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=counts,
            mode='lines+markers',
            name='Yearly',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=10, color=self.colors['secondary']),
            hovertemplate='<b>Year %{x}</b><br>Tracks added: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            **self.base_layout,
            title='Songs Added Over Time',
            xaxis_title='Year',
            yaxis_title='Number of Songs',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_mood_distribution(self, mood_dist):
        """Pie chart of mood distribution"""
        moods = list(mood_dist.keys())
        percentages = [mood_dist[m]['percentage'] for m in moods]
        
        # Color mapping for moods
        mood_colors = {
            'Happy': '#FFD700',
            'Sad': '#4169E1',
            'Energetic': '#FF6347',
            'Chill': '#7FFFD4'
        }
        
        colors = [mood_colors.get(m, self.colors['primary']) for m in moods]
        
        fig = go.Figure(data=[
            go.Pie(
                labels=moods,
                values=percentages,
                marker=dict(colors=colors, line=dict(color='#000000', width=2)),
                textinfo='label+percent',
                textfont=dict(size=16, color='white'),
                hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            **self.base_layout,
            title='Your Mood Distribution',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_popularity_distribution(self, dist):
        """Bar chart of popularity classification"""
        classes = ['Low', 'Medium', 'High']
        counts = [dist.get(c, {}).get('count', 0) for c in classes]
        percentages = [dist.get(c, {}).get('percentage', 0) for c in classes]
        
        colors_map = {
            'Low': '#8B008B',
            'Medium': '#FF8C00',
            'High': '#FFD700'
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=classes,
                y=counts,
                marker=dict(
                    color=[colors_map[c] for c in classes],
                    line=dict(color='white', width=2)
                ),
                text=[f"{p:.1f}%" for p in percentages],
                textposition='outside',
                hovertemplate='<b>%{x} Popularity</b><br>Count: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            **self.base_layout,
            title='Popularity Distribution',
            xaxis_title='Popularity Class',
            yaxis_title='Number of Tracks',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_mood_radar(self, mood_dist):
        """Radar plot showing mood profile"""
        moods = ['Happy', 'Sad', 'Energetic', 'Chill']
        values = [mood_dist.get(m, {}).get('percentage', 0) for m in moods]
        
        # Close the radar by repeating first value
        moods_closed = moods + [moods[0]]
        values_closed = values + [values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=moods_closed,
            fill='toself',
            fillcolor='rgba(29, 185, 84, 0.3)',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=10, color=self.colors['secondary']),
            name='Your Mood Profile',
            hovertemplate='<b>%{theta}</b><br>%{r:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            **self.base_layout,
            title='Your Mood Radar',
            polar=dict(
                bgcolor=self.colors['card'],
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values) * 1.2 if values else 100],
                    gridcolor=self.colors['text_secondary'],
                    tickfont=dict(color=self.colors['text'])
                ),
                angularaxis=dict(
                    gridcolor=self.colors['text_secondary'],
                    tickfont=dict(color=self.colors['text'], size=14)
                )
            ),
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_audio_features_radar(self, stats):
        """Radar plot of audio features (placeholder - needs feature data)"""
        # This would need actual audio feature data from API
        # For now, create a basic radar with available stats
        
        features = ['Popularity', 'Explicit', 'Duration']
        values = [
            stats.get('popularity', {}).get('average', 50),
            stats.get('explicit', {}).get('percentage', 0),
            min(stats.get('total_duration', {}).get('hours', 0) * 5, 100)  # Normalize
        ]
        
        features_closed = features + [features[0]]
        values_closed = values + [values[0]]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=features_closed,
            fill='toself',
            fillcolor='rgba(29, 185, 84, 0.3)',
            line=dict(color=self.colors['primary'], width=3),
            marker=dict(size=10, color=self.colors['secondary'])
        ))
        
        fig.update_layout(
            **self.base_layout,
            title='Your Audio Feature Profile',
            polar=dict(
                bgcolor=self.colors['card'],
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor=self.colors['text_secondary']
                ),
                angularaxis=dict(
                    gridcolor=self.colors['text_secondary'],
                    tickfont=dict(size=14)
                )
            ),
            height=500
        )
        
        return fig